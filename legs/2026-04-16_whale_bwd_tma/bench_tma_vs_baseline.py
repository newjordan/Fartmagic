"""A/B TMA fused_delta vs 3-kernel baseline inside one process."""
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
    p.add_argument("--shape", default="4,4096,8,4,64")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--variants", default="baseline,fused_delta,fused_delta_tma")
    a = p.parse_args()
    B, T, H, KV, D = [int(x) for x in a.shape.split(",")]
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = torch.randn_like(q)

    variants = a.variants.split(",")
    # Let fused_delta / fused_delta_tma run at all T for this bench.
    os.environ["WHALE_FUSED_DELTA_T_MAX"] = "1048576"

    for variant in variants:
        os.environ["WHALE_BWD_VARIANT"] = variant
        for _ in range(a.warm):
            q.grad = k.grad = v.grad = None
            o = whale_attn_fast(q, k, v, True)
            o.backward(g)
    torch.cuda.synchronize()

    rs = {v: [] for v in variants}
    for r in range(a.rounds):
        order = list(variants) if r % 2 == 0 else list(reversed(variants))
        for variant in order:
            ms = bench(variant, q, k, v, g, a.iters)
            rs[variant].append(ms)

    print(f"shape=({B},{T},{H},{KV},{D}) rounds={a.rounds} iters={a.iters}")
    base_trim = []
    for variant in variants:
        xs = rs[variant]
        cut = a.rounds // 6
        trimmed = sorted(xs)[cut: -cut if cut else None]
        m = statistics.fmean(trimmed)
        if variant == variants[0]:
            base_trim = m
        print(
            f"  {variant:20s} mean={m:.4f}ms  median={statistics.median(xs):.4f}ms  "
            f"min={min(xs):.4f}ms  stdev={statistics.stdev(xs):.4f}ms"
        )
    print()
    for variant in variants[1:]:
        xs = rs[variant]
        cut = a.rounds // 6
        m = statistics.fmean(sorted(xs)[cut: -cut if cut else None])
        delta = (m - base_trim) / base_trim * 100.0
        print(f"  {variant} vs {variants[0]}: {m - base_trim:+.4f}ms  ({delta:+.2f}%)")


if __name__ == "__main__":
    main()
