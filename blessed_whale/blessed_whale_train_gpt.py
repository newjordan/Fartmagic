"""Blessed Whale runner.

Uses the preserved PR1493-family donor trainer contract from `pr1493_decompressed.py`
while replacing FA3 attention with the blessed whale native kernel path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

try:
    _dynamo_disable = torch.compiler.disable
except AttributeError:
    try:
        import torch._dynamo as _torch_dynamo
    except Exception:
        def _dynamo_disable(fn):
            return fn
    else:
        _dynamo_disable = _torch_dynamo.disable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_attn_interface import flash_attn_func as fa3_attn_fwd
from vault.whale_kernel_triton import custom_whale_attn_fwd, whale_fwd_fa3_bwd

CONTRACT_TRAINER_NAME = "pr1493_decompressed.py"
FORBIDDEN_TRAINER_NAME = "pr1493_train_gpt.py"
BASE_TRAINER_PATH = REPO_ROOT / CONTRACT_TRAINER_NAME

BLESSED_WHALE_DEFAULTS = {
    "WHALE_BWD_VARIANT": "fused_delta_tma",
    "WHALE_FUSED_DELTA_T_MAX": "3072",
    "WHALE_BWD_Q_CONFIG": "128,64,8,5",
    "WHALE_BWD_Q_MAXNREG": "1984",
    "WHALE_BWD_KV_TMA_CONFIG": "128,64,8,3",
    "WHALE_BWD_KV_TMA_MAXNREG": "256",
}


def _install_kernel_env_defaults() -> None:
    for key, value in BLESSED_WHALE_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _configure_dynamo() -> None:
    try:
        import torch._dynamo.config as dynamo_config
    except Exception:
        return

    recompile_limit = max(8, int(os.environ.get("BLESSED_WHALE_RECOMPILE_LIMIT", "64")))
    accumulated_limit = max(
        recompile_limit,
        int(os.environ.get("BLESSED_WHALE_ACCUMULATED_RECOMPILE_LIMIT", str(recompile_limit * 4))),
    )

    if hasattr(dynamo_config, "recompile_limit"):
        dynamo_config.recompile_limit = recompile_limit
    if hasattr(dynamo_config, "cache_size_limit"):
        dynamo_config.cache_size_limit = max(int(getattr(dynamo_config, "cache_size_limit", recompile_limit)), recompile_limit)
    if hasattr(dynamo_config, "accumulated_recompile_limit"):
        dynamo_config.accumulated_recompile_limit = accumulated_limit
    if hasattr(dynamo_config, "accumulated_cache_size_limit"):
        dynamo_config.accumulated_cache_size_limit = max(
            int(getattr(dynamo_config, "accumulated_cache_size_limit", accumulated_limit)),
            accumulated_limit,
        )
    if hasattr(dynamo_config, "fail_on_recompile_limit_hit"):
        dynamo_config.fail_on_recompile_limit_hit = False

    print(
        "blessed_whale_dynamo:"
        f" recompile_limit={getattr(dynamo_config, 'recompile_limit', 'na')}"
        f" accumulated_recompile_limit={getattr(dynamo_config, 'accumulated_recompile_limit', 'na')}"
        f" fail_on_recompile_limit_hit={getattr(dynamo_config, 'fail_on_recompile_limit_hit', 'na')}"
    )


def _resolve_attention_mode() -> str:
    mode = os.environ.get("BLESSED_WHALE_ATTN_MODE", os.environ.get("WH_KERN_ATTN_MODE", "whale_native")).strip().lower()
    if mode not in {"whale_native", "fa3", "hybrid"}:
        raise ValueError(f"unsupported attention mode={mode!r}")
    return mode


def _load_base():
    resolved = BASE_TRAINER_PATH.resolve()
    if resolved.name != CONTRACT_TRAINER_NAME:
        raise RuntimeError(f"unexpected donor trainer path: {resolved}")
    if FORBIDDEN_TRAINER_NAME in str(resolved):
        raise RuntimeError(f"forbidden donor trainer path resolved: {resolved}")
    spec = importlib.util.spec_from_file_location("blessed_whale_pr1493_base", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load donor trainer: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_hybrid_warmstart_calls() -> int:
    return max(0, int(os.environ.get("BLESSED_WHALE_HYBRID_WARMSTART_CALLS", "0")))


def _resolve_startup_fa3_only() -> bool:
    return bool(int(os.environ.get("BLESSED_WHALE_STARTUP_FA3_ONLY", "0")))


def _resolve_loop_warmup_fa3_only() -> bool:
    return bool(int(os.environ.get("BLESSED_WHALE_LOOP_WARMUP_FA3_ONLY", "0")))


def _resolve_eval_fa3_only() -> bool:
    return bool(int(os.environ.get("BLESSED_WHALE_EVAL_FA3_ONLY", "0")))


def _resolve_reset_warmstart_on_loop_enable() -> bool:
    return bool(int(os.environ.get("BLESSED_WHALE_RESET_WARMSTART_ON_LOOP_ENABLE", "1")))


def _resolve_hybrid_backend() -> str:
    raw = os.environ.get("BLESSED_WHALE_HYBRID_BACKEND", "whale_native_bwd").strip().lower()
    aliases = {
        "legacy": "fa3_bwd",
        "whale_fa3_bwd": "fa3_bwd",
        "whale_native": "whale_native_bwd",
        "whale": "whale_native_bwd",
    }
    backend = aliases.get(raw, raw)
    if backend not in {"whale_native_bwd", "fa3_bwd", "fa3"}:
        raise ValueError(
            "unsupported BLESSED_WHALE_HYBRID_BACKEND="
            f"{raw!r}; expected one of whale_native_bwd, fa3_bwd, fa3"
        )
    return backend


def _resolve_compile_fullgraph() -> bool:
    return bool(
        int(
            os.environ.get(
                "BLESSED_WHALE_TORCH_COMPILE_FULLGRAPH",
                os.environ.get("TORCH_COMPILE_FULLGRAPH", "1"),
            )
        )
    )


def _hybrid_requires_dynamic_dispatch() -> bool:
    return any(
        (
            _resolve_hybrid_warmstart_calls() > 0,
            _resolve_startup_fa3_only(),
            _resolve_loop_warmup_fa3_only(),
            _resolve_eval_fa3_only(),
        )
    )


def _configure_compile_mode(attn_mode: str) -> None:
    if (
        attn_mode == "hybrid"
        and "BLESSED_WHALE_TORCH_COMPILE_FULLGRAPH" not in os.environ
        and "TORCH_COMPILE_FULLGRAPH" not in os.environ
    ):
        if _hybrid_requires_dynamic_dispatch():
            os.environ["BLESSED_WHALE_TORCH_COMPILE_FULLGRAPH"] = "0"
            reason = (
                "hybrid_warmstart_dispatch"
                if _resolve_hybrid_warmstart_calls() > 0
                else "hybrid_dispatch"
            )
            print(f"blessed_whale_compile: fullgraph=0 reason={reason}")
        else:
            print("blessed_whale_compile: fullgraph=1 reason=hybrid_static")


class _HybridDispatch:
    def __init__(self) -> None:
        self.warmstart_calls = _resolve_hybrid_warmstart_calls()
        self.hybrid_backend = _resolve_hybrid_backend()
        self.startup_fa3_only = _resolve_startup_fa3_only()
        self.loop_warmup_fa3_only = _resolve_loop_warmup_fa3_only()
        self.eval_fa3_only = _resolve_eval_fa3_only()
        self.reset_warmstart_on_loop_enable = _resolve_reset_warmstart_on_loop_enable()
        self.startup_complete = not self.startup_fa3_only
        self.loop_warmup_active = False
        self.calls = 0

    def blessed_whale_on_startup_complete(self) -> None:
        self.startup_complete = True
        if self.startup_fa3_only:
            self.calls = 0

    def blessed_whale_on_loop_warmup_start(self) -> None:
        self.loop_warmup_active = True

    def blessed_whale_on_loop_warmup_end(self) -> None:
        self.loop_warmup_active = False

    def blessed_whale_on_loop_enable(self) -> None:
        if self.reset_warmstart_on_loop_enable and self.warmstart_calls > 0:
            self.calls = 0

    @_dynamo_disable
    def __call__(self, q, k, v, *args, **kwargs):
        if self.eval_fa3_only and not torch.is_grad_enabled():
            return fa3_attn_fwd(q, k, v, *args, **kwargs)
        if self.loop_warmup_fa3_only and self.loop_warmup_active:
            return fa3_attn_fwd(q, k, v, *args, **kwargs)
        if self.startup_fa3_only and not self.startup_complete:
            return fa3_attn_fwd(q, k, v, *args, **kwargs)
        if self.warmstart_calls > 0:
            self.calls += 1
            if self.calls <= self.warmstart_calls:
                return fa3_attn_fwd(q, k, v, *args, **kwargs)
        if self.hybrid_backend == "fa3_bwd":
            return whale_fwd_fa3_bwd(q, k, v, *args, **kwargs)
        if self.hybrid_backend == "fa3":
            return fa3_attn_fwd(q, k, v, *args, **kwargs)
        return custom_whale_attn_fwd(q, k, v, *args, **kwargs)


def _make_hybrid_dispatch():
    return _HybridDispatch()


def _resolve_hybrid_static_callable():
    hybrid_backend = _resolve_hybrid_backend()
    if hybrid_backend == "fa3_bwd":
        return whale_fwd_fa3_bwd
    if hybrid_backend == "fa3":
        return fa3_attn_fwd
    return custom_whale_attn_fwd


def _maybe_warn_shape_mismatch() -> None:
    model_dim = int(os.environ.get("MODEL_DIM", "576"))
    num_heads = int(os.environ.get("NUM_HEADS", "8"))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", "4"))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", "2048"))
    donor_head_dim = model_dim // num_heads
    blessed_head_dim = int(os.environ.get("BLESSED_WHALE_BLESSED_HEAD_DIM", str(donor_head_dim)))
    if donor_head_dim != blessed_head_dim:
        print(
            "blessed_whale_warning: donor attention shape differs from blessed kernel evidence"
            f" donor_shape=2,{train_seq_len},{num_heads},{num_kv_heads},{donor_head_dim}"
            f" blessed_shape={os.environ.get('BLESSED_WHALE_BLESSED_BENCH_SHAPE', 'unknown')}"
            " recommendation=rerank kernel on donor shape or prefer hybrid diagnostic before native promotion"
        )


def main() -> None:
    _install_kernel_env_defaults()
    _configure_dynamo()
    attn_mode = _resolve_attention_mode()
    _configure_compile_mode(attn_mode)
    base = _load_base()
    _maybe_warn_shape_mismatch()
    if attn_mode == "whale_native":
        base.flash_attn_3_func = custom_whale_attn_fwd
    elif attn_mode == "hybrid":
        warmstart_calls = _resolve_hybrid_warmstart_calls()
        hybrid_backend = _resolve_hybrid_backend()
        if warmstart_calls > 0:
            print(
                "blessed_whale_hybrid_warmstart:"
                f" calls={warmstart_calls}"
                " mode=fa3_then_hybrid"
            )
        print(f"blessed_whale_hybrid_backend: mode={hybrid_backend}")
        if _resolve_loop_warmup_fa3_only():
            print("blessed_whale_hybrid_loop_warmup: mode=fa3_only")
        if _resolve_eval_fa3_only():
            print("blessed_whale_hybrid_eval: mode=fa3_only")
        if _resolve_reset_warmstart_on_loop_enable() and warmstart_calls > 0:
            print("blessed_whale_hybrid_loop_enable: reset_warmstart=1")
        if _hybrid_requires_dynamic_dispatch():
            print("blessed_whale_hybrid_dispatch: mode=dynamic")
            base.flash_attn_3_func = _make_hybrid_dispatch()
        else:
            print("blessed_whale_hybrid_dispatch: mode=static")
            base.flash_attn_3_func = _resolve_hybrid_static_callable()
    print(
        "blessed_whale:"
        f" benchmark={os.environ.get('BLESSED_WHALE_BENCHMARK_ID', 'unknown')}"
        f" trainer={CONTRACT_TRAINER_NAME}"
        f" profile={os.environ.get('BLESSED_WHALE_PROFILE_ID', 'unknown')}"
        f" donor=whale_l12_dim576_matrixbits5"
        f" attn_mode={attn_mode}"
        f" family={os.environ.get('BLESSED_WHALE_FAMILY_ID', 'unknown')}"
        f" dq={os.environ.get('WHALE_BWD_Q_CONFIG')}"
        f" qmr={os.environ.get('WHALE_BWD_Q_MAXNREG')}"
        f" dkdv={os.environ.get('WHALE_BWD_KV_TMA_CONFIG')}"
        f" dkdv_mr={os.environ.get('WHALE_BWD_KV_TMA_MAXNREG')}"
        f" variant={os.environ.get('WHALE_BWD_VARIANT')}"
        f" source={os.environ.get('BLESSED_WHALE_SOURCE_EVIDENCE', 'unknown')}"
    )
    print(
        "blessed_whale_compile:"
        f" fullgraph={int(_resolve_compile_fullgraph())}"
        f" disable_torch_compile={os.environ.get('DISABLE_TORCH_COMPILE', '0')}"
    )
    base.main()


if __name__ == "__main__":
    main()
