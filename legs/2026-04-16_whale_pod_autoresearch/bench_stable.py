"""Stable head-to-head: warms up the GPU thoroughly, interleaves backend calls
to average out thermal / L2 state, reports high-iter means with confidence
intervals.

Each (shape, backend) is re-measured many times with interleaving; the report
is mean +/- stdev across the rounds.
"""
from __future__ import annotations

import argparse, json, os, statistics, time
import torch
import torch.nn.functional as F

from vault.whale_kernel_triton import (
    custom_whale_attn_fwd,
    whale_attn_fast,
    whale_fwd_fa3_bwd,
)


def _have_fa3():
    try:
        from flash_attn_interface import flash_attn_func
        return True
    except Exception:
        return False


def sdpa_call(q, k, v, causal=True):
    B, T, H, D = q.shape
    KV = k.shape[2]; rep = H // KV
    kk = k.repeat_interleave(rep, dim=2) if rep != 1 else k
    vv = v.repeat_interleave(rep, dim=2) if rep != 1 else v
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), is_causal=causal,
    )
    return out.transpose(1, 2)


def fa3_call(q, k, v, causal=True):
    from flash_attn_interface import flash_attn_func
    r = flash_attn_func(q, k, v, causal=causal)
    return r[0] if isinstance(r, tuple) else r


def whale_call(q, k, v, causal=True):
    return custom_whale_attn_fwd(q, k, v, causal=causal)


def whale_fast_call(q, k, v, causal=True):
    return whale_attn_fast(q, k, v, causal=causal)


def whale_hybrid_call(q, k, v, causal=True):
    return whale_fwd_fa3_bwd(q, k, v, causal=causal)


BACKENDS = {"whale": whale_call, "whale_fast": whale_fast_call,
            "whale_hybrid": whale_hybrid_call,
            "sdpa": sdpa_call, "fa3": fa3_call}


def _new_inputs(B, T, H, KV, D, grad=False, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((B, T, H, D), generator=g, device="cuda", dtype=dtype, requires_grad=grad)
    k = torch.randn((B, T, KV, D), generator=g, device="cuda", dtype=dtype, requires_grad=grad)
    v = torch.randn((B, T, KV, D), generator=g, device="cuda", dtype=dtype, requires_grad=grad)
    return q, k, v


def time_events(fn, iters):
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    torch.cuda.synchronize()
    for a, b in events:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in events]


def thorough_warmup(fns, iters=200):
    """Call each fn many times to fully JIT + autotune, then warm the GPU."""
    for f in fns:
        for _ in range(iters):
            f()
    torch.cuda.synchronize()


def measure(shape, backends, rounds=5, iters=100, fwd_bwd=True):
    B, T, H, KV, D = shape
    # Build inputs once per backend; outputs comparable.
    q_nograd, k_nograd, v_nograd = _new_inputs(B, T, H, KV, D, grad=False)
    q_grad, k_grad, v_grad = _new_inputs(B, T, H, KV, D, grad=True)
    # Create a gout with fa3 call (any call works for shape)
    with torch.no_grad():
        sample_out = BACKENDS[backends[0]](q_nograd, k_nograd, v_nograd, causal=True)
    gout = torch.randn_like(sample_out)

    # Prebuild callables
    fwd_calls = {b: (lambda b=b: BACKENDS[b](q_nograd, k_nograd, v_nograd, causal=True)) for b in backends}
    def fb_call(b):
        q_grad.grad = k_grad.grad = v_grad.grad = None
        out = BACKENDS[b](q_grad, k_grad, v_grad, causal=True)
        out.backward(gout)
    fb_calls = {b: (lambda b=b: fb_call(b)) for b in backends}

    # Heavy warmup on all
    thorough_warmup(list(fwd_calls.values()), iters=50)
    if fwd_bwd:
        thorough_warmup(list(fb_calls.values()), iters=50)

    results = {b: {"fwd_rounds": [], "fb_rounds": []} for b in backends}

    for _ in range(rounds):
        for b in backends:
            t = time_events(fwd_calls[b], iters)
            results[b]["fwd_rounds"].append(statistics.mean(t))
            if fwd_bwd:
                t = time_events(fb_calls[b], iters)
                results[b]["fb_rounds"].append(statistics.mean(t))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="stable")
    ap.add_argument("--shape", default="4,2048,8,4,64")
    ap.add_argument("--shapes", default=None, help="semicolon-separated list of shapes")
    ap.add_argument("--backends", default="whale,fa3,sdpa")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    else:
        shapes = [tuple(int(x) for x in args.shape.split(","))]
    backends = [b for b in args.backends.split(",") if b in BACKENDS]
    if "fa3" in backends and not _have_fa3():
        backends.remove("fa3")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    payload = {"label": args.label, "device": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "env_fwd": os.environ.get("WHALE_FWD_CONFIG"),
               "env_bkv": os.environ.get("WHALE_BWD_KV_CONFIG"),
               "env_bq":  os.environ.get("WHALE_BWD_Q_CONFIG"),
               "rounds": args.rounds, "iters_per_round": args.iters, "results": []}
    for shape in shapes:
        r = measure(shape, backends, rounds=args.rounds, iters=args.iters)
        print(f"\n== {args.label}  shape={shape}")
        entry = {"shape": shape, "backends": {}}
        for b in backends:
            fwd = r[b]["fwd_rounds"]; fb = r[b]["fb_rounds"]
            entry["backends"][b] = {
                "fwd_mean_ms": statistics.mean(fwd), "fwd_std_ms": statistics.stdev(fwd) if len(fwd)>1 else 0,
                "fb_mean_ms": statistics.mean(fb), "fb_std_ms": statistics.stdev(fb) if len(fb)>1 else 0,
            }
            print(f"  {b:>12}  fwd={statistics.mean(fwd):.3f}+-{statistics.stdev(fwd) if len(fwd)>1 else 0:.3f}ms "
                  f"  fwd+bwd={statistics.mean(fb):.3f}+-{statistics.stdev(fb) if len(fb)>1 else 0:.3f}ms")
        payload["results"].append(entry)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
