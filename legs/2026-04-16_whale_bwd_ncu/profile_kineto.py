"""Per-kernel GPU-time profile for whale vs FA3 on the headline shape.

Uses torch.profiler (kineto) to attribute GPU time to individual kernels.
Also dumps Triton kernel static metadata (n_regs, shared_mem, n_warps).
"""
import argparse, os, sys, json
from collections import defaultdict

import torch


def kineto_profile(fn, steps: int = 20, warmup: int = 8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(steps):
            fn()
            torch.cuda.synchronize()

    # Aggregate CUDA kernel times
    agg: dict[str, dict] = defaultdict(lambda: {"us": 0.0, "calls": 0})
    for e in prof.events():
        # Only true CUDA kernel events (device_type == 1 in kineto)
        if getattr(e, "device_type", None) is None:
            continue
        dt = getattr(e, "device_type", None)
        if str(dt) != "DeviceType.CUDA":
            continue
        name = e.name
        # strip launch args/clutter
        short = name.split("<")[0]
        agg[short]["us"] += (e.cuda_time_total or e.self_device_time_total or 0) / max(steps, 1)
        agg[short]["calls"] += 1
    # sort by total time
    rows = sorted(agg.items(), key=lambda kv: -kv[1]["us"])
    for name, s in rows[:20]:
        print(f"  {s['us']:9.2f} us  calls/iter={s['calls'] / steps:5.2f}  {name[:120]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["whale", "fa3"], default="whale")
    ap.add_argument("--shape", default="4,2048,8,4,64")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--include-fwd", action="store_true")
    args = ap.parse_args()

    B, T, H, KV, D = [int(x) for x in args.shape.split(",")]
    os.environ.setdefault("WHALE_BWD_VARIANT", "fused_delta")

    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = torch.randn_like(q)

    if args.backend == "whale":
        from vault.whale_kernel_triton import whale_attn_fast
        def step():
            q.grad = k.grad = v.grad = None
            o = whale_attn_fast(q, k, v, True)
            o.backward(g)
    else:
        try:
            from flash_attn_interface import flash_attn_func
        except ImportError:
            from flash_attn import flash_attn_func
        def step():
            q.grad = k.grad = v.grad = None
            o = flash_attn_func(q, k, v, causal=True)
            if isinstance(o, tuple):
                o = o[0]
            o.backward(g)

    print(f"=== {args.backend} backend, shape={args.shape} ===")
    kineto_profile(step, steps=args.steps)


if __name__ == "__main__":
    main()
