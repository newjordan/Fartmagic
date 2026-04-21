"""wh_kern_int runner.

Uses the preserved PR1493-family donor trainer contract from `pr1493_decompressed.py`
while replacing FA3 attention with the blessed whale native kernel path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from vault.whale_kernel_triton import custom_whale_attn_fwd, whale_fwd_fa3_bwd

BASE_TRAINER_PATH = Path(__file__).resolve().parents[1] / "pr1493_decompressed.py"

BLESSED_KERNEL_DEFAULTS = {
    "WHALE_BWD_VARIANT": "fused_delta_tma",
    "WHALE_FUSED_DELTA_T_MAX": "3072",
    "WHALE_BWD_Q_CONFIG": "128,64,8,5",
    "WHALE_BWD_Q_MAXNREG": "1984",
    "WHALE_BWD_KV_TMA_CONFIG": "128,64,8,3",
    "WHALE_BWD_KV_TMA_MAXNREG": "256",
}


def _install_kernel_env_defaults() -> None:
    for key, value in BLESSED_KERNEL_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _resolve_attention_mode():
    mode = os.environ.get("WH_KERN_ATTN_MODE", "whale_native").strip().lower()
    if mode not in {"whale_native", "fa3", "hybrid"}:
        raise ValueError(f"unsupported WH_KERN_ATTN_MODE={mode!r}")
    return mode


def _load_base():
    spec = importlib.util.spec_from_file_location("wh_kern_int_pr1493_base", BASE_TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load donor trainer: {BASE_TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _install_kernel_env_defaults()
    attn_mode = _resolve_attention_mode()
    base = _load_base()
    if attn_mode == "whale_native":
        base.flash_attn_3_func = custom_whale_attn_fwd
    elif attn_mode == "hybrid":
        base.flash_attn_3_func = whale_fwd_fa3_bwd
    print(
        "wh_kern_int: donor=pr1493_decompressed"
        f" attn_mode={attn_mode}"
        f" dq={os.environ.get('WHALE_BWD_Q_CONFIG')}"
        f" qmr={os.environ.get('WHALE_BWD_Q_MAXNREG')}"
        f" dkdv={os.environ.get('WHALE_BWD_KV_TMA_CONFIG')}"
        f" dkdv_mr={os.environ.get('WHALE_BWD_KV_TMA_MAXNREG')}"
        f" variant={os.environ.get('WHALE_BWD_VARIANT')}"
        f" source={os.environ.get('WH_KERN_WINNER_SOURCE_EVIDENCE', 'unknown')}"
    )
    base.main()


if __name__ == "__main__":
    main()
