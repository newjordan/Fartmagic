"""find_winner.py — probe autotune winners for _attn_bwd_dkdv_kernel.

Usage (on pod, after clearing the triton cache so autotune re-runs):
    rm -rf /root/.triton/cache
    export WHALE_BWD_VARIANT=baseline
    export TRITON_PRINT_AUTOTUNING=1
    CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python3 \
        legs/2026-04-17_whale_dkdv_autotune_expand/find_winner.py \
        2>&1 | tee legs/2026-04-17_whale_dkdv_autotune_expand/logs/find_winner_$(date +%Y%m%d_%H%M%S).log

Notes:
- Triton prints its autotune table to stderr. We capture both streams above.
- We dispatch one fwd + one bwd per shape via whale_attn_fast (which triggers
  _attn_bwd_dkdv_kernel along the baseline path from
  vault/whale_kernel_triton.py:1558). Autotune runs once per shape/signature,
  prints the winning config, caches the compiled kernel.
- No timing. This is a tuner probe, not a bench.
- Single-GPU only (see CLAUDE.md memory single_gpu_testing).
"""
from __future__ import annotations

import os
import sys
import torch

# Surface autotune picks even if caller forgot to export it.
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")
# Baseline path = the one that hits _bwd_kv_configs() (expanded to 56).
os.environ.setdefault("WHALE_BWD_VARIANT", "baseline")

from vault.whale_kernel_triton import whale_attn_fast, _bwd_kv_configs  # noqa: E402


SHAPES = [
    # (B, T, H, KV, D)
    (2, 8192, 8, 4, 64),
    (2, 4096, 8, 4, 64),
    (2, 2048, 8, 4, 64),
    (1, 16384, 8, 4, 64),
]


def main() -> int:
    if not torch.cuda.is_available():
        print("[find_winner] CUDA unavailable", file=sys.stderr)
        return 2
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    dtype = torch.bfloat16

    n_cfg = len(_bwd_kv_configs())
    print(f"[find_winner] _bwd_kv_configs count = {n_cfg}", flush=True)
    if n_cfg != 56:
        print(f"[find_winner][FATAL] expected 56 configs, got {n_cfg}",
              file=sys.stderr)
        return 3

    for (B, T, H, KV, D) in SHAPES:
        print("", flush=True)
        print(f"==== shape B={B} T={T} H={H} KV={KV} D={D} ====", flush=True)
        # Fresh tensors per shape; autotune keys on the signature so each new
        # (T, D) triggers a new tuning pass with TRITON_PRINT_AUTOTUNING output.
        q = torch.randn(B, T, H,  D, device=dev, dtype=dtype, requires_grad=True)
        k = torch.randn(B, T, KV, D, device=dev, dtype=dtype, requires_grad=True)
        v = torch.randn(B, T, KV, D, device=dev, dtype=dtype, requires_grad=True)
        # fwd+bwd to force both dkdv and dq autotune to fire.
        out = whale_attn_fast(q, k, v, causal=True)
        g = torch.randn_like(out)
        out.backward(g)
        torch.cuda.synchronize()
        # Free grads so the next shape starts clean.
        q.grad = k.grad = v.grad = None
        del q, k, v, out, g
        torch.cuda.empty_cache()

    print("", flush=True)
    print("[find_winner] done. Grep the log for 'Triton autotuning' or"
          " 'Autotuning kernel' to see the winners.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
