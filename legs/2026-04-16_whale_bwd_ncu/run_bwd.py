"""Minimal runner for ncu profiling of whale backward kernels.

Launches a fixed number of fwd+bwd iterations after a warmup phase, on a
single headline shape, so ncu --launch-count targets only the measured kernels.
"""
import argparse, os, sys, torch

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["whale", "fa3"], default="whale")
    ap.add_argument("--variant", default="fused_delta")
    ap.add_argument("--shape", default="4,2048,8,4,64")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    B, T, H, KV, D = [int(x) for x in args.shape.split(",")]
    os.environ.setdefault("WHALE_BWD_VARIANT", args.variant)

    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = torch.randn_like(q)

    if args.backend == "whale":
        from vault.whale_kernel_triton import whale_attn_fast
        def fn():
            return whale_attn_fast(q, k, v, True)
    else:
        try:
            from flash_attn_interface import flash_attn_func  # FA3
        except ImportError:
            from flash_attn import flash_attn_func
        def fn():
            return flash_attn_func(q, k, v, causal=True)

    # Warmup (outside the measured launch window)
    for _ in range(args.warmup):
        q.grad = k.grad = v.grad = None
        o = fn()
        if isinstance(o, tuple):
            o = o[0]
        o.backward(g)
    torch.cuda.synchronize()

    # Measured iterations
    torch.cuda.nvtx.range_push("MEASURED")
    for _ in range(args.iters):
        q.grad = k.grad = v.grad = None
        o = fn()
        if isinstance(o, tuple):
            o = o[0]
        o.backward(g)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print(f"done: backend={args.backend} variant={args.variant} shape={args.shape}")


if __name__ == "__main__":
    main()
