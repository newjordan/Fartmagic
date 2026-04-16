"""Tight A/B benchmark of fused_delta vs fused_delta_tma inside one process.
Alternates variants round-by-round to cancel thermal drift, uses CUDA events
for sub-microsecond wall timing.
"""
import argparse
import os
import statistics

import torch

from vault.whale_kernel_triton import whale_attn_fast


def bench(variant: str, q, k, v, g, iters: int) -> float:
    os.environ["WHALE_BWD_VARIANT"] = variant
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        q.grad = k.grad = v.grad = None
        o = whale_attn_fast(q, k, v, True)
        o.backward(g)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="4,2048,8,4,64")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--warm", type=int, default=30)
    a = p.parse_args()
    B, T, H, KV, D = [int(x) for x in a.shape.split(",")]
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = torch.randn_like(q)

    # warm-up both variants (autotune + caches)
    for variant in ("fused_delta", "fused_delta_tma"):
        os.environ["WHALE_BWD_VARIANT"] = variant
        for _ in range(a.warm):
            q.grad = k.grad = v.grad = None
            o = whale_attn_fast(q, k, v, True)
            o.backward(g)
    torch.cuda.synchronize()

    rs = {"fused_delta": [], "fused_delta_tma": []}
    for r in range(a.rounds):
        # alternate order across rounds to cancel drift
        order = ("fused_delta", "fused_delta_tma") if r % 2 == 0 else ("fused_delta_tma", "fused_delta")
        for variant in order:
            ms = bench(variant, q, k, v, g, a.iters)
            rs[variant].append(ms)

    print(f"shape=({B},{T},{H},{KV},{D}) rounds={a.rounds} iters={a.iters}")
    for variant, xs in rs.items():
        xs_sorted = sorted(xs)
        trimmed = xs_sorted[a.rounds // 6: -(a.rounds // 6) if a.rounds // 6 else None]
        print(
            f"  {variant:20s} mean={statistics.fmean(trimmed):.4f}ms  "
            f"median={statistics.median(xs):.4f}ms  min={min(xs):.4f}ms  "
            f"stdev={statistics.stdev(xs):.4f}ms  n={len(xs)}"
        )
    fd = statistics.fmean(sorted(rs["fused_delta"])[a.rounds // 6: -(a.rounds // 6) if a.rounds // 6 else None])
    tma = statistics.fmean(sorted(rs["fused_delta_tma"])[a.rounds // 6: -(a.rounds // 6) if a.rounds // 6 else None])
    delta = (tma - fd) / fd * 100.0
    print(f"  tma vs fused_delta: {tma - fd:+.4f}ms  ({delta:+.2f}%)")


if __name__ == "__main__":
    main()
