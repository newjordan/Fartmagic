"""Per-kernel GPU-time profile for whale (auto) vs FA3 at long T.
Adapted from legs/2026-04-16_whale_bwd_ncu/profile_kineto.py.
"""
import argparse, os, json, sys
from collections import defaultdict
import torch


def kineto_profile(fn, steps=20, warmup=8):
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

    agg = defaultdict(lambda: {"us": 0.0, "calls": 0})
    for e in prof.events():
        if str(getattr(e, "device_type", None)) != "DeviceType.CUDA":
            continue
        short = e.name.split("<")[0]
        agg[short]["us"] += (e.cuda_time_total or e.self_device_time_total or 0) / max(steps, 1)
        agg[short]["calls"] += 1
    rows = sorted(agg.items(), key=lambda kv: -kv[1]["us"])
    out = []
    for name, s in rows[:25]:
        out.append({"name": name, "us_per_iter": s["us"], "calls_per_iter": s["calls"] / steps})
        print(f"  {s['us']:9.2f} us  calls/iter={s['calls']/steps:5.2f}  {name[:120]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["whale", "fa3"], required=True)
    ap.add_argument("--shape", default="2,8192,8,4,64")
    ap.add_argument("--variant", default="auto", help="WHALE_BWD_VARIANT")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    B, T, H, KV, D = [int(x) for x in args.shape.split(",")]
    os.environ["WHALE_BWD_VARIANT"] = args.variant

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
        from flash_attn_interface import flash_attn_func
        def step():
            q.grad = k.grad = v.grad = None
            o = flash_attn_func(q, k, v, causal=True)
            if isinstance(o, tuple):
                o = o[0]
            o.backward(g)

    print(f"=== {args.backend} variant={args.variant} shape={args.shape} ===")
    rows = kineto_profile(step, steps=args.steps)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"backend": args.backend, "variant": args.variant,
                       "shape": [B,T,H,KV,D], "steps": args.steps, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
