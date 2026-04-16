"""Autoresearch head-to-head: whale vs SDPA vs FA3.

bf16, causal, GQA supported. Measures fwd-only and fwd+bwd latency with
CUDA events, plus numerical errors vs SDPA reference.
"""
from __future__ import annotations

import argparse, json, os, statistics, time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from vault.whale_kernel_triton import custom_whale_attn_fwd


def _have_fa3():
    try:
        from flash_attn_interface import flash_attn_func  # noqa
        return True
    except Exception:
        return False


def sdpa_call(q, k, v, causal=True):
    # q,k,v: [B,T,H,D] / [B,T,KV,D]; SDPA wants [B,H,T,D] and MQA expansion.
    B, T, H, D = q.shape
    KV = k.shape[2]
    rep = H // KV
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


BACKENDS = {"whale": whale_call, "sdpa": sdpa_call, "fa3": fa3_call}


def _new_inputs(B, T, H, KV, D, device="cuda", dtype=torch.bfloat16, seed=0, grad=True):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((B, T, H, D), generator=g, device=device, dtype=dtype, requires_grad=grad)
    k = torch.randn((B, T, KV, D), generator=g, device=device, dtype=dtype, requires_grad=grad)
    v = torch.randn((B, T, KV, D), generator=g, device=device, dtype=dtype, requires_grad=grad)
    return q, k, v


def _cuda_time_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for a, b in events:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    times = [a.elapsed_time(b) for a, b in events]
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def _err_vs_sdpa(out, q, k, v, gout):
    # Reference: SDPA with same inputs
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    ref = sdpa_call(q_, k_, v_, causal=True)
    ref_out = ref.float()
    fwd_err = (out.float() - ref_out).abs().max().item()
    ref.backward(gout)
    return fwd_err, q_.grad.detach().float(), k_.grad.detach().float(), v_.grad.detach().float()


def measure_shape(B, T, H, KV, D, backend, dtype=torch.bfloat16, warmup=5, iters=20, seed=0):
    fn = BACKENDS[backend]
    q, k, v = _new_inputs(B, T, H, KV, D, dtype=dtype, seed=seed, grad=False)

    # FWD only
    fwd = _cuda_time_ms(lambda: fn(q, k, v, causal=True), warmup, iters)

    # FWD+BWD (fresh grad-tracked inputs)
    qg, kg, vg = _new_inputs(B, T, H, KV, D, dtype=dtype, seed=seed, grad=True)
    gout = torch.randn_like(fn(qg, kg, vg, causal=True))

    def step():
        qg.grad = kg.grad = vg.grad = None
        out = fn(qg, kg, vg, causal=True)
        out.backward(gout)

    fb = _cuda_time_ms(step, warmup, iters)

    # Numerical error vs SDPA reference (only if not sdpa itself)
    err = {"fwd_max_abs": None, "dq_max_abs": None, "dk_max_abs": None, "dv_max_abs": None}
    if backend != "sdpa":
        out = fn(qg, kg, vg, causal=True)
        qg.grad = kg.grad = vg.grad = None
        out.backward(gout, retain_graph=False)
        ref_fwd_err, ref_dq, ref_dk, ref_dv = _err_vs_sdpa(out.detach(), qg.detach(), kg.detach(), vg.detach(), gout)
        err["fwd_max_abs"] = ref_fwd_err
        err["dq_max_abs"] = (qg.grad.detach().float() - ref_dq).abs().max().item()
        err["dk_max_abs"] = (kg.grad.detach().float() - ref_dk).abs().max().item()
        err["dv_max_abs"] = (vg.grad.detach().float() - ref_dv).abs().max().item()

    return {
        "backend": backend, "shape": {"B": B, "T": T, "H": H, "KV": KV, "D": D},
        "dtype": str(dtype), "fwd": fwd, "fwd_bwd": fb, "err": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="pod_bench")
    ap.add_argument("--shapes", default="8,2048,8,4,64")  # semicolons separate shapes
    ap.add_argument("--backends", default="whale,sdpa,fa3")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    shape_strs = [s.strip() for s in args.shapes.replace(";", "|").split("|") if s.strip()]
    shapes = [tuple(int(x) for x in s.split(",")) for s in shape_strs]
    backends = [b for b in args.backends.split(",") if b.strip() in BACKENDS]
    if "fa3" in backends and not _have_fa3():
        print("[warn] fa3 not available, dropping"); backends.remove("fa3")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    all_results = []
    for shp in shapes:
        for be in backends:
            try:
                r = measure_shape(*shp, be, warmup=args.warmup, iters=args.iters)
                print(f"{be:>5} shape={shp} fwd={r['fwd']['mean_ms']:.3f}ms fwd+bwd={r['fwd_bwd']['mean_ms']:.3f}ms "
                      f"fwd_err={r['err']['fwd_max_abs']}")
                all_results.append(r)
            except Exception as e:
                print(f"FAIL {be} {shp}: {type(e).__name__}: {e}")
                all_results.append({"backend": be, "shape": {"B": shp[0], "T": shp[1], "H": shp[2], "KV": shp[3], "D": shp[4]}, "error": f"{type(e).__name__}: {e}"})

    out = {
        "label": args.label,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "fa3_available": _have_fa3(),
        "warmup": args.warmup, "iters": args.iters,
        "results": all_results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
