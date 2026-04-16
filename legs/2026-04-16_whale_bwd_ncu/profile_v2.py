"""Verify GPU kernel time via multiple methods:
1. torch.profiler kineto with key_averages
2. CUDA events wrapping the fwd+bwd (event_fwd_bwd)
3. CUDA events wrapping only the bwd pass
4. Direct kernel enumeration via prof.events()
"""
import argparse, os, torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["whale", "fa3"], default="whale")
    ap.add_argument("--shape", default="4,2048,8,4,64")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    B, T, H, KV, D = [int(x) for x in args.shape.split(",")]
    os.environ.setdefault("WHALE_BWD_VARIANT", "fused_delta")

    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, T, KV, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = torch.randn_like(q)

    if args.backend == "whale":
        from vault.whale_kernel_triton import whale_attn_fast
        def fwd():
            return whale_attn_fast(q, k, v, True)
    else:
        try:
            from flash_attn_interface import flash_attn_func
        except ImportError:
            from flash_attn import flash_attn_func
        def fwd():
            o = flash_attn_func(q, k, v, causal=True)
            return o[0] if isinstance(o, tuple) else o

    # Warmup
    for _ in range(10):
        q.grad = k.grad = v.grad = None
        o = fwd(); o.backward(g)
    torch.cuda.synchronize()

    # Method 1: kineto events with per-event enumeration
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(args.steps):
            q.grad = k.grad = v.grad = None
            o = fwd(); o.backward(g)
        torch.cuda.synchronize()

    print(f"=== {args.backend} kineto key_averages (sorted by self_device_time_total) ===")
    ka = prof.key_averages().table(
        sort_by="self_device_time_total", row_limit=15,
        top_level_events_only=False,
    )
    print(ka)

    # Sum all device kernels manually to compare with wall
    total_device_us = 0.0
    n_events = 0
    for e in prof.events():
        t = getattr(e, "self_device_time_total", 0) or 0
        if t > 0:
            total_device_us += t
            n_events += 1
    print(f"[{args.backend}] total self_device_time across {n_events} kernel events = "
          f"{total_device_us:.1f} us -> per-iter = {total_device_us / args.steps:.2f} us")

    # Method 2: event-based fwd+bwd and bwd-only
    N = 200
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(N):
        q.grad = k.grad = v.grad = None
        o = fwd(); o.backward(g)
    e1.record(); torch.cuda.synchronize()
    fwd_bwd_us = e0.elapsed_time(e1) * 1000 / N
    print(f"[{args.backend}] event fwd+bwd = {fwd_bwd_us:.2f} us/iter")

    # bwd-only: precompute o once, time only backward
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(N):
        q.grad = k.grad = v.grad = None
        o = fwd()
    e1.record(); torch.cuda.synchronize()
    fwd_us = e0.elapsed_time(e1) * 1000 / N
    print(f"[{args.backend}] event fwd only = {fwd_us:.2f} us/iter (bwd = {fwd_bwd_us - fwd_us:.2f} us)")


if __name__ == "__main__":
    main()
