#!/usr/bin/env python3
"""Numerics + micro-speed harness for whale attention.

Compares `vault.whale_kernel_triton.custom_whale_attn_fwd` against PyTorch's
`scaled_dot_product_attention` across a small sweep of (head_dim, GQA ratio, T).
Also measures forward-only and forward+backward wall time per case.

Writes a JSON file with every measurement so RESULTS.md can cite it directly.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def sdpa_whale(q, k, v, causal=True):
    q2 = q.transpose(1, 2)
    k2 = k.transpose(1, 2)
    v2 = v.transpose(1, 2)
    if k2.size(1) != q2.size(1):
        rep = q2.size(1) // k2.size(1)
        k2 = k2.repeat_interleave(rep, dim=1)
        v2 = v2.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    return out.transpose(1, 2)


def time_ms(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def make_qkv(B, T, H, KV, D, requires_grad=False, device="cuda", dtype=torch.bfloat16):
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, T, KV, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, T, KV, D, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def run_case(whale_fn, B, T, H, KV, D, *, bwd: bool):
    q, k, v = make_qkv(B, T, H, KV, D, requires_grad=bwd)
    qr, kr, vr = (q.detach().clone().requires_grad_(bwd),
                  k.detach().clone().requires_grad_(bwd),
                  v.detach().clone().requires_grad_(bwd))
    with torch.no_grad() if not bwd else torch.enable_grad():
        whale_out = whale_fn(q, k, v, causal=True)
        sdpa_out = sdpa_whale(qr, kr, vr, causal=True)
    err_fwd = max_abs(whale_out, sdpa_out)
    case = {
        "B": B, "T": T, "H": H, "KV": KV, "D": D,
        "fwd_max_abs_err": err_fwd,
    }
    if bwd:
        do = torch.randn_like(whale_out)
        (whale_out * do).sum().backward()
        (sdpa_out * do).sum().backward()
        case["dq_max_abs_err"] = max_abs(q.grad, qr.grad)
        case["dk_max_abs_err"] = max_abs(k.grad, kr.grad)
        case["dv_max_abs_err"] = max_abs(v.grad, vr.grad)
    return case


def run_speed(whale_fn, B, T, H, KV, D):
    q, k, v = make_qkv(B, T, H, KV, D, requires_grad=False)
    fwd_only = time_ms(lambda: whale_fn(q, k, v, causal=True))
    sdpa_fwd = time_ms(lambda: sdpa_whale(q, k, v, causal=True))
    q2, k2, v2 = make_qkv(B, T, H, KV, D, requires_grad=True)
    def fwd_bwd_whale():
        out = whale_fn(q2, k2, v2, causal=True)
        (out * 1.0).sum().backward()
        q2.grad = None; k2.grad = None; v2.grad = None
    q3, k3, v3 = make_qkv(B, T, H, KV, D, requires_grad=True)
    def fwd_bwd_sdpa():
        out = sdpa_whale(q3, k3, v3, causal=True)
        (out * 1.0).sum().backward()
        q3.grad = None; k3.grad = None; v3.grad = None
    fwd_bwd_w = time_ms(fwd_bwd_whale, warmup=2, iters=10)
    fwd_bwd_s = time_ms(fwd_bwd_sdpa, warmup=2, iters=10)
    return {
        "B": B, "T": T, "H": H, "KV": KV, "D": D,
        "whale_fwd": fwd_only,
        "sdpa_fwd": sdpa_fwd,
        "whale_fwd_bwd": fwd_bwd_w,
        "sdpa_fwd_bwd": fwd_bwd_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="vault.whale_kernel_triton:custom_whale_attn_fwd",
                    help="module:callable spec for the whale kernel under test")
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--label", default="current", help="label for this run")
    ap.add_argument("--skip-bwd", action="store_true", help="skip backward numerics + bwd speed")
    args = ap.parse_args()

    mod_name, attr = args.kernel.split(":", 1)
    import importlib
    whale_fn = getattr(importlib.import_module(mod_name), attr)

    torch.manual_seed(1337)
    shapes = [
        # B, T, H, KV, D
        (2, 256, 8, 4, 64),
        (2, 1024, 8, 4, 64),
        (2, 2048, 8, 4, 64),
        (2, 1024, 8, 8, 64),    # MHA
        (2, 1024, 8, 2, 64),    # GQA 4:1
        (2, 1024, 4, 4, 128),   # D=128
        (2, 1024, 8, 4, 32),    # D=32
    ]

    numerics = []
    for (B, T, H, KV, D) in shapes:
        numerics.append(run_case(whale_fn, B, T, H, KV, D, bwd=not args.skip_bwd))

    speed = []
    for (B, T, H, KV, D) in [(8, 2048, 8, 4, 64), (4, 1024, 8, 4, 64)]:
        speed.append(run_speed(whale_fn, B, T, H, KV, D))

    payload = {
        "label": args.label,
        "kernel": args.kernel,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "numerics": numerics,
        "speed": speed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
