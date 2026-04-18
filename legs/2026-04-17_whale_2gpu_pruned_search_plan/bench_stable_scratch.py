"""Stable head-to-head bench that imports a leg-local whale kernel copy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics

import torch
import torch.nn.functional as F

from frozen_vault.whale_kernel_triton import (
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
    bsz, seqlen, heads, dim = q.shape
    kv_heads = k.shape[2]
    rep = heads // kv_heads
    kk = k.repeat_interleave(rep, dim=2) if rep != 1 else k
    vv = v.repeat_interleave(rep, dim=2) if rep != 1 else v
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), is_causal=causal
    )
    return out.transpose(1, 2)


def fa3_call(q, k, v, causal=True):
    from flash_attn_interface import flash_attn_func

    out = flash_attn_func(q, k, v, causal=causal)
    return out[0] if isinstance(out, tuple) else out


def whale_call(q, k, v, causal=True):
    return custom_whale_attn_fwd(q, k, v, causal=causal)


def whale_fast_call(q, k, v, causal=True):
    return whale_attn_fast(q, k, v, causal=causal)


def whale_hybrid_call(q, k, v, causal=True):
    return whale_fwd_fa3_bwd(q, k, v, causal=causal)


BACKENDS = {
    'whale': whale_call,
    'whale_fast': whale_fast_call,
    'whale_hybrid': whale_hybrid_call,
    'sdpa': sdpa_call,
    'fa3': fa3_call,
}


def _new_inputs(bsz, seqlen, heads, kv_heads, dim, grad=False, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn((bsz, seqlen, heads, dim), generator=g, device='cuda', dtype=dtype, requires_grad=grad)
    k = torch.randn((bsz, seqlen, kv_heads, dim), generator=g, device='cuda', dtype=dtype, requires_grad=grad)
    v = torch.randn((bsz, seqlen, kv_heads, dim), generator=g, device='cuda', dtype=dtype, requires_grad=grad)
    return q, k, v


def time_events(fn, iters):
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    torch.cuda.synchronize()
    for start, stop in events:
        start.record()
        fn()
        stop.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(stop) for start, stop in events]


def thorough_warmup(fns, iters=200):
    for fn in fns:
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()


def measure(shape, backends, rounds=5, iters=100, fwd_bwd=True):
    bsz, seqlen, heads, kv_heads, dim = shape
    q_nograd, k_nograd, v_nograd = _new_inputs(bsz, seqlen, heads, kv_heads, dim, grad=False)
    q_grad, k_grad, v_grad = _new_inputs(bsz, seqlen, heads, kv_heads, dim, grad=True)
    with torch.no_grad():
        sample_out = BACKENDS[backends[0]](q_nograd, k_nograd, v_nograd, causal=True)
    gout = torch.randn_like(sample_out)

    fwd_calls = {name: (lambda name=name: BACKENDS[name](q_nograd, k_nograd, v_nograd, causal=True)) for name in backends}

    def fb_call(name):
        q_grad.grad = None
        k_grad.grad = None
        v_grad.grad = None
        out = BACKENDS[name](q_grad, k_grad, v_grad, causal=True)
        out.backward(gout)

    fb_calls = {name: (lambda name=name: fb_call(name)) for name in backends}

    thorough_warmup(list(fwd_calls.values()), iters=50)
    if fwd_bwd:
        thorough_warmup(list(fb_calls.values()), iters=50)

    results = {name: {'fwd_rounds': [], 'fb_rounds': []} for name in backends}
    for _ in range(rounds):
        for name in backends:
            fwd = time_events(fwd_calls[name], iters)
            results[name]['fwd_rounds'].append(statistics.mean(fwd))
            if fwd_bwd:
                fb = time_events(fb_calls[name], iters)
                results[name]['fb_rounds'].append(statistics.mean(fb))
    return results


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--label', default='stable')
    ap.add_argument('--shape', default='4,2048,8,4,64')
    ap.add_argument('--shapes', default=None)
    ap.add_argument('--backends', default='whale,fa3,sdpa')
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--iters', type=int, default=100)
    args = ap.parse_args()

    if args.shapes:
        shapes = [tuple(int(x) for x in shape.split(',')) for shape in args.shapes.split(';')]
    else:
        shapes = [tuple(int(x) for x in args.shape.split(','))]
    backends = [name for name in args.backends.split(',') if name in BACKENDS]
    if 'fa3' in backends and not _have_fa3():
        backends.remove('fa3')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    kernel_path = os.path.join(os.path.dirname(__file__), 'frozen_vault', 'whale_kernel_triton.py')
    payload = {
        'label': args.label,
        'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'kernel_path': kernel_path,
        'kernel_sha256': sha256sum(kernel_path),
        'env_fwd': os.environ.get('WHALE_FWD_CONFIG'),
        'env_bwd_variant': os.environ.get('WHALE_BWD_VARIANT'),
        'env_bkv_tma': os.environ.get('WHALE_BWD_KV_TMA_CONFIG'),
        'env_bkv_tma_maxnreg': os.environ.get('WHALE_BWD_KV_TMA_MAXNREG'),
        'env_bq': os.environ.get('WHALE_BWD_Q_CONFIG'),
        'env_bq_maxnreg': os.environ.get('WHALE_BWD_Q_MAXNREG'),
        'rounds': args.rounds,
        'iters_per_round': args.iters,
        'results': [],
    }
    for shape in shapes:
        result = measure(shape, backends, rounds=args.rounds, iters=args.iters)
        print(f"\n== {args.label}  shape={shape}")
        entry = {'shape': shape, 'backends': {}}
        for name in backends:
            fwd = result[name]['fwd_rounds']
            fb = result[name]['fb_rounds']
            entry['backends'][name] = {
                'fwd_mean_ms': statistics.mean(fwd),
                'fwd_std_ms': statistics.stdev(fwd) if len(fwd) > 1 else 0,
                'fb_mean_ms': statistics.mean(fb),
                'fb_std_ms': statistics.stdev(fb) if len(fb) > 1 else 0,
            }
            print(
                f"  {name:>12}  fwd={statistics.mean(fwd):.3f}+-{statistics.stdev(fwd) if len(fwd) > 1 else 0:.3f}ms"
                f"   fwd+bwd={statistics.mean(fb):.3f}+-{statistics.stdev(fb) if len(fb) > 1 else 0:.3f}ms"
            )
        payload['results'].append(entry)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)


if __name__ == '__main__':
    main()
