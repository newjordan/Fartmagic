#!/usr/bin/env python3
"""D72 whale donor + FA3, clean 100-step loop resume control."""

from __future__ import annotations

import json
import os
import re
import selectors
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path


TEST = {
    "test_id": "2026-04-26_d72_loop_transition_fa3_clean100",
    "date": "2026-04-26",
    "hypothesis": "Run a matching 100-step FA3-only loop-transition control from the same D72 loop-onset checkpoint so the tail16dot 100-step probe has a same-horizon throughput baseline.",
    "parent_baseline": "FA3 clean24 reached 819661 tok/s at step 479; tail16dot clean24 reached 770756 tok/s and tail16dot clean100 is the active stability probe.",
    "parent_benchmark_id": "2026-04-26_d72_split64_tail16dot_loop_transition_probe100",
    "standard_label": "mechanics_4x_diagnostic_resume_probe_clean_stop",
    "gpu_count": 4,
    "build_seconds": 1200,
    "eval_seconds": 1200,
    "trainer_wallclock_seconds": 0,
    "size_cap_bytes": 16_000_000,
}

MODEL = {
    "donor_lineage": "whale_l12_dim576_matrixbits5_d72",
    "trainer": "blessed_whale/blessed_whale_train_gpt.py",
    "contract_trainer": "pr1493_decompressed.py",
    "seed": 444,
    "iterations": 20_000,
    "vocab_size": 8192,
    "tokenizer_path": "./data/tokenizers/fineweb_8192_bpe.model",
    "dataset_dir": "./data/datasets/fineweb10B_sp8192",
    "num_layers": 12,
    "model_dim": 576,
    "embedding_dim": 576,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 72,
    "xsa_last_n": 11,
    "rope_dims": 16,
    "matrix_bits": 5,
    "matrix_clip_sigmas": 12.85,
    "qk_gain_init": 5.25,
    "mlp_mult": 4.0,
    "matrix_lr": 0.022,
    "muon_momentum": 0.99,
    "muon_wd": 0.095,
    "ema_decay": 0.9965,
    "warmdown_frac": 0.72,
    "parallel_residual_start": 8,
}

KERNEL = {
    "benchmark_id": "RUNPY_D72_LOOP_TRANSITION_FA3_CLEAN100",
    "description": "D72 whale donor FA3-only clean 100-step loop-transition control",
    "purpose": "provide a same-horizon FA3 control for the tail16dot 100-step loop-transition probe",
    "attn_mode": "fa3",
    "hybrid_backend": "fa3",
    "kernel_profile_id": "fa3_control_no_custom_kernel",
    "family_id": "fa3_control",
    "dq_tile": "128,64,8,5",
    "qmr": 1984,
    "dkdv_tile": "128,64,8,3",
    "dkdv_maxnreg": 256,
    "bwd_variant": "fused_delta_tma",
    "fused_delta_t_max": 3072,
    "source_evidence": "/workspace/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_primary_validate_dq_128_64_8_5_dkdv_128_64_8_3_mr_256_qmr_1984_20260421_000841.json",
    "blessed_head_dim": 72,
    "blessed_bench_shape": "2,2048,8,4,72",
    "torch_compile_fullgraph": 1,
    "disable_torch_compile": 0,
    "hybrid_warmstart_calls": 0,
    "reset_warmstart_on_loop_enable": 0,
    "startup_fa3_only": 0,
    "loop_warmup_fa3_only": 0,
    "eval_fa3_only": 0,
}

LOOPS = {
    "num_loops": 2,
    "loop_start": 3,
    "loop_end": 5,
    "enable_looping_at": 0.35,
    "loop_warmup_do_backward": 1,
    "loop_warmup_do_optimizer_step": 1,
    "loop_warmup_qk_gain_scale": 1.0,
}

EVAL = {
    "skip_post_train": 1,
    "gptq_reserve_seconds": 0,
    "sliding_window_enabled": 0,
    "ttt_enabled": 0,
    "ttt_lr": 0.005,
    "ttt_epochs": 3,
    "ttt_momentum": 0.9,
    "gptq_calibration_batches": 64,
    "compressor": "brotli",
    "train_log_every": 1,
    "val_loss_every": 0,
}

ARTIFACTS = {
    "loop_onset_checkpoint_path": "",
    "loop_onset_stop_after_save": 0,
    "resume_checkpoint_path": "legs/2026-04-26_d72_loop_onset_checkpoint_builder/checkpoints/loop_onset.pt",
    "resume_probe_steps": 100,
    "resume_probe_stop_after": 1,
}

RESULT_HEADER = (
    "test_id\tdate\tgpu_count\tbuild_seconds\teval_seconds\tsize_cap_bytes\t"
    "raw_step\traw_bpb\tpost_ema_bpb\tquant_bpb\tsliding_bpb\t"
    "ttt_bpb\t"
    "total_bytes\tdecision\tlog_path\trun_py\n"
)

RESOLVED_ENV_KEYS = {
    "PATH",
    "RUN_ID",
    "SEED",
    "NPROC_PER_NODE",
    "ITERATIONS",
    "MAX_WALLCLOCK_SECONDS",
    "TRAIN_LOG_EVERY",
    "VAL_LOSS_EVERY",
    "GPTQ_RESERVE_SECONDS",
    "SKIP_POST_TRAIN",
    "SLIDING_WINDOW_ENABLED",
    "TTT_ENABLED",
    "TTT_LR",
    "TTT_EPOCHS",
    "TTT_MOMENTUM",
    "GPTQ_CALIBRATION_BATCHES",
    "COMPRESSOR",
    "DATA_DIR",
    "VOCAB_SIZE",
    "NUM_LAYERS",
    "MODEL_DIM",
    "EMBEDDING_DIM",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "XSA_LAST_N",
    "ROPE_DIMS",
    "MATRIX_BITS",
    "MATRIX_CLIP_SIGMAS",
    "QK_GAIN_INIT",
    "MLP_MULT",
    "MATRIX_LR",
    "MUON_MOMENTUM",
    "MUON_WD",
    "EMA_DECAY",
    "WARMDOWN_FRAC",
    "PARALLEL_RESIDUAL_START",
    "NUM_LOOPS",
    "LOOP_START",
    "LOOP_END",
    "ENABLE_LOOPING_AT",
    "LOOP_WARMUP_DO_BACKWARD",
    "LOOP_WARMUP_DO_OPTIM_STEP",
    "LOOP_WARMUP_QK_GAIN_SCALE",
    "BLESSED_WHALE_BENCHMARK_ID",
    "BLESSED_WHALE_DESCRIPTION",
    "BLESSED_WHALE_PURPOSE",
    "BLESSED_WHALE_PROFILE_ID",
    "BLESSED_WHALE_TRAINER_CONTRACT",
    "BLESSED_WHALE_ATTN_MODE",
    "BLESSED_WHALE_HYBRID_BACKEND",
    "BLESSED_WHALE_FAMILY_ID",
    "BLESSED_WHALE_DQ_TILE",
    "BLESSED_WHALE_QMR",
    "BLESSED_WHALE_DKDV_TILE",
    "BLESSED_WHALE_DKDV_MAXNREG",
    "BLESSED_WHALE_SOURCE_EVIDENCE",
    "BLESSED_WHALE_BLESSED_HEAD_DIM",
    "BLESSED_WHALE_BLESSED_BENCH_SHAPE",
    "BLESSED_WHALE_TORCH_COMPILE_FULLGRAPH",
    "DISABLE_TORCH_COMPILE",
    "BLESSED_WHALE_HYBRID_WARMSTART_CALLS",
    "BLESSED_WHALE_RESET_WARMSTART_ON_LOOP_ENABLE",
    "BLESSED_WHALE_STARTUP_FA3_ONLY",
    "BLESSED_WHALE_LOOP_WARMUP_FA3_ONLY",
    "BLESSED_WHALE_EVAL_FA3_ONLY",
    "WH_KERN_ATTN_MODE",
    "WH_KERN_HYBRID_BACKEND",
    "WHALE_BWD_VARIANT",
    "WHALE_FUSED_DELTA_T_MAX",
    "WHALE_BWD_Q_CONFIG",
    "WHALE_BWD_Q_MAXNREG",
    "WHALE_BWD_KV_TMA_CONFIG",
    "WHALE_BWD_KV_TMA_MAXNREG",
    "PROFILE_ENABLED",
    "LOOP_ONSET_CHECKPOINT_PATH",
    "LOOP_ONSET_STOP_AFTER_SAVE",
    "RESUME_CHECKPOINT_PATH",
    "RESUME_PROBE_STEPS",
    "RESUME_PROBE_STOP_AFTER",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def leg_dir() -> Path:
    return Path(__file__).resolve().parent


def make_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    run_id = TEST["test_id"] + "_s" + str(MODEL["seed"])
    updates = {
        "PATH": "/venv/main/bin:" + env.get("PATH", ""),
        "RUN_ID": run_id,
        "SEED": str(MODEL["seed"]),
        "NPROC_PER_NODE": str(TEST["gpu_count"]),
        "ITERATIONS": str(MODEL["iterations"]),
        "MAX_WALLCLOCK_SECONDS": str(TEST["trainer_wallclock_seconds"]),
        "TRAIN_LOG_EVERY": str(EVAL["train_log_every"]),
        "VAL_LOSS_EVERY": str(EVAL["val_loss_every"]),
        "GPTQ_RESERVE_SECONDS": str(EVAL["gptq_reserve_seconds"]),
        "SKIP_POST_TRAIN": str(EVAL["skip_post_train"]),
        "SLIDING_WINDOW_ENABLED": str(EVAL["sliding_window_enabled"]),
        "TTT_ENABLED": str(EVAL["ttt_enabled"]),
        "TTT_LR": str(EVAL["ttt_lr"]),
        "TTT_EPOCHS": str(EVAL["ttt_epochs"]),
        "TTT_MOMENTUM": str(EVAL["ttt_momentum"]),
        "GPTQ_CALIBRATION_BATCHES": str(EVAL["gptq_calibration_batches"]),
        "COMPRESSOR": EVAL["compressor"],
        "DATA_DIR": "./data",
        "VOCAB_SIZE": str(MODEL["vocab_size"]),
        "NUM_LAYERS": str(MODEL["num_layers"]),
        "MODEL_DIM": str(MODEL["model_dim"]),
        "EMBEDDING_DIM": str(MODEL["embedding_dim"]),
        "NUM_HEADS": str(MODEL["num_heads"]),
        "NUM_KV_HEADS": str(MODEL["num_kv_heads"]),
        "XSA_LAST_N": str(MODEL["xsa_last_n"]),
        "ROPE_DIMS": str(MODEL["rope_dims"]),
        "MATRIX_BITS": str(MODEL["matrix_bits"]),
        "MATRIX_CLIP_SIGMAS": str(MODEL["matrix_clip_sigmas"]),
        "QK_GAIN_INIT": str(MODEL["qk_gain_init"]),
        "MLP_MULT": str(MODEL["mlp_mult"]),
        "MATRIX_LR": str(MODEL["matrix_lr"]),
        "MUON_MOMENTUM": str(MODEL["muon_momentum"]),
        "MUON_WD": str(MODEL["muon_wd"]),
        "EMA_DECAY": str(MODEL["ema_decay"]),
        "WARMDOWN_FRAC": str(MODEL["warmdown_frac"]),
        "PARALLEL_RESIDUAL_START": str(MODEL["parallel_residual_start"]),
        "NUM_LOOPS": str(LOOPS["num_loops"]),
        "LOOP_START": str(LOOPS["loop_start"]),
        "LOOP_END": str(LOOPS["loop_end"]),
        "ENABLE_LOOPING_AT": str(LOOPS["enable_looping_at"]),
        "LOOP_WARMUP_DO_BACKWARD": str(LOOPS["loop_warmup_do_backward"]),
        "LOOP_WARMUP_DO_OPTIM_STEP": str(LOOPS["loop_warmup_do_optimizer_step"]),
        "LOOP_WARMUP_QK_GAIN_SCALE": str(LOOPS["loop_warmup_qk_gain_scale"]),
        "BLESSED_WHALE_BENCHMARK_ID": KERNEL["benchmark_id"],
        "BLESSED_WHALE_DESCRIPTION": KERNEL["description"],
        "BLESSED_WHALE_PURPOSE": KERNEL["purpose"],
        "BLESSED_WHALE_PROFILE_ID": KERNEL["kernel_profile_id"],
        "BLESSED_WHALE_TRAINER_CONTRACT": MODEL["contract_trainer"],
        "BLESSED_WHALE_ATTN_MODE": KERNEL["attn_mode"],
        "BLESSED_WHALE_HYBRID_BACKEND": KERNEL["hybrid_backend"],
        "BLESSED_WHALE_FAMILY_ID": KERNEL["family_id"],
        "BLESSED_WHALE_DQ_TILE": KERNEL["dq_tile"],
        "BLESSED_WHALE_QMR": str(KERNEL["qmr"]),
        "BLESSED_WHALE_DKDV_TILE": KERNEL["dkdv_tile"],
        "BLESSED_WHALE_DKDV_MAXNREG": str(KERNEL["dkdv_maxnreg"]),
        "BLESSED_WHALE_SOURCE_EVIDENCE": KERNEL["source_evidence"],
        "BLESSED_WHALE_BLESSED_HEAD_DIM": str(KERNEL["blessed_head_dim"]),
        "BLESSED_WHALE_BLESSED_BENCH_SHAPE": KERNEL["blessed_bench_shape"],
        "BLESSED_WHALE_TORCH_COMPILE_FULLGRAPH": str(KERNEL["torch_compile_fullgraph"]),
        "DISABLE_TORCH_COMPILE": str(KERNEL["disable_torch_compile"]),
        "BLESSED_WHALE_HYBRID_WARMSTART_CALLS": str(KERNEL["hybrid_warmstart_calls"]),
        "BLESSED_WHALE_RESET_WARMSTART_ON_LOOP_ENABLE": str(KERNEL["reset_warmstart_on_loop_enable"]),
        "BLESSED_WHALE_STARTUP_FA3_ONLY": str(KERNEL["startup_fa3_only"]),
        "BLESSED_WHALE_LOOP_WARMUP_FA3_ONLY": str(KERNEL["loop_warmup_fa3_only"]),
        "BLESSED_WHALE_EVAL_FA3_ONLY": str(KERNEL["eval_fa3_only"]),
        "WH_KERN_ATTN_MODE": KERNEL["attn_mode"],
        "WH_KERN_HYBRID_BACKEND": KERNEL["hybrid_backend"],
        "WHALE_BWD_VARIANT": KERNEL["bwd_variant"],
        "WHALE_FUSED_DELTA_T_MAX": str(KERNEL["fused_delta_t_max"]),
        "WHALE_BWD_Q_CONFIG": KERNEL["dq_tile"],
        "WHALE_BWD_Q_MAXNREG": str(KERNEL["qmr"]),
        "WHALE_BWD_KV_TMA_CONFIG": KERNEL["dkdv_tile"],
        "WHALE_BWD_KV_TMA_MAXNREG": str(KERNEL["dkdv_maxnreg"]),
        "PROFILE_ENABLED": "0",
        "LOOP_ONSET_CHECKPOINT_PATH": ARTIFACTS["loop_onset_checkpoint_path"],
        "LOOP_ONSET_STOP_AFTER_SAVE": str(ARTIFACTS["loop_onset_stop_after_save"]),
        "RESUME_CHECKPOINT_PATH": ARTIFACTS["resume_checkpoint_path"],
        "RESUME_PROBE_STEPS": str(ARTIFACTS["resume_probe_steps"]),
        "RESUME_PROBE_STOP_AFTER": str(ARTIFACTS["resume_probe_stop_after"]),
    }
    if MODEL["model_dim"] // MODEL["num_heads"] != MODEL["head_dim"]:
        raise RuntimeError("D72 shape invariant failed")
    if set(updates) != RESOLVED_ENV_KEYS:
        unexpected = sorted(set(updates) - RESOLVED_ENV_KEYS)
        missing = sorted(RESOLVED_ENV_KEYS - set(updates))
        raise RuntimeError(f"resolved env key mismatch unexpected={unexpected} missing={missing}")
    env.update(updates)
    return env


def preflight(root: Path) -> None:
    required = [
        root / MODEL["trainer"],
        root / MODEL["contract_trainer"],
        root / MODEL["tokenizer_path"],
        root / MODEL["dataset_dir"],
        root / "vault/whale_kernel_triton.py",
        root / "scripts/pod_stack_guard.sh",
    ]
    if ARTIFACTS["resume_checkpoint_path"]:
        required.append(root / ARTIFACTS["resume_checkpoint_path"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("FATAL: missing required paths:\n" + "\n".join(missing))
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("FATAL: nvidia-smi not found")
    if not Path("/venv/main/bin/torchrun").is_file():
        raise SystemExit("FATAL: /venv/main/bin/torchrun not found")
    active = subprocess.run(
        ["pgrep", "-af", r"[t]orchrun|[b]lessed_whale_train_gpt|[p]r1493_decompressed"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if active.returncode == 0 and active.stdout.strip():
        raise SystemExit("FATAL: training already running:\n" + active.stdout)
    guard = subprocess.run(
        ["bash", "scripts/pod_stack_guard.sh", "verify"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if guard.returncode != 0:
        raise SystemExit("FATAL: pod stack guard failed:\n" + guard.stdout + guard.stderr)


def resolved_config(root: Path, log_path: Path, result_path: Path) -> dict[str, object]:
    return {
        "test": TEST,
        "model": MODEL,
        "kernel": KERNEL,
        "loops": LOOPS,
        "eval": EVAL,
        "artifacts": ARTIFACTS,
        "repo_root": str(root),
        "run_py": str(Path(__file__).resolve()),
        "log_path": str(log_path),
        "result_path": str(result_path),
    }


def stream_run(root: Path, env: dict[str, str], log_path: Path) -> int:
    cmd = [
        "/venv/main/bin/torchrun",
        "--standalone",
        "--nproc_per_node",
        str(TEST["gpu_count"]),
        str(root / MODEL["trainer"]),
    ]
    timeout = int(TEST["build_seconds"]) + int(TEST["eval_seconds"])
    start = time.time()

    def terminate_process_group(proc: subprocess.Popen[str], sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                if time.time() - start > timeout:
                    log.write(f"TIMEOUT: exceeded total mechanics budget {timeout}s\n")
                    log.flush()
                    terminate_process_group(proc, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        terminate_process_group(proc, signal.SIGKILL)
                        proc.wait(timeout=30)
                    return 124
                events = selector.select(timeout=1.0)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        sys.stdout.write(line)
                        log.write(line)
                        log.flush()
                if proc.poll() is not None:
                    tail = proc.stdout.read()
                    if tail:
                        sys.stdout.write(tail)
                        log.write(tail)
                        log.flush()
                    return proc.returncode if proc.returncode is not None else 1
        finally:
            selector.unregister(proc.stdout)
            selector.close()
            proc.stdout.close()


def last_match(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else "NA"


def parse_result(log_text: str, rc: int) -> dict[str, str]:
    raw_step = last_match(r"^(\d+)/20000 val_loss: [0-9.]+ val_bpb: [0-9.]+$", log_text)
    raw_bpb = last_match(r"^\d+/20000 val_loss: [0-9.]+ val_bpb: ([0-9.]+)$", log_text)
    post_ema = last_match(r"pre-quantization post-ema val_loss:[0-9.]+ val_bpb:([0-9.]+)", log_text)
    quant = last_match(r"^quantized val_loss:[0-9.]+ val_bpb:([0-9.]+)", log_text)
    sliding = last_match(r"^quantized_sliding_window val_loss:[0-9.]+ val_bpb:([0-9.]+)", log_text)
    ttt = last_match(r"^quantized_ttt val_loss:[0-9.]+ val_bpb:([0-9.]+)", log_text)
    total_bytes = last_match(r"Total submission size [^:]+: ([0-9]+) bytes", log_text)
    checkpoint_saved = "loop_onset_checkpoint:saved" in log_text and "stopping_early: loop_onset_checkpoint" in log_text
    resume_probe_complete = "stopping_early: resume_probe_complete" in log_text
    if rc != 0:
        decision = f"fail_rc_{rc}"
    elif resume_probe_complete:
        decision = "resume_probe_complete"
    elif checkpoint_saved:
        decision = "checkpoint_saved_loop_onset"
    elif total_bytes != "NA" and int(total_bytes) > int(TEST["size_cap_bytes"]):
        decision = "fail_over_16mb"
    elif "NA" in {post_ema, quant, sliding, ttt, total_bytes}:
        decision = "scout_missing_posttrain"
    else:
        decision = "complete_under_cap" if int(total_bytes) <= int(TEST["size_cap_bytes"]) else "fail_over_16mb"
    return {
        "raw_step": raw_step,
        "raw_bpb": raw_bpb,
        "post_ema_bpb": post_ema,
        "quant_bpb": quant,
        "sliding_bpb": sliding,
        "ttt_bpb": ttt,
        "total_bytes": total_bytes,
        "decision": decision,
    }


def append_result(result_path: Path, log_path: Path, parsed: dict[str, str]) -> None:
    if not result_path.exists():
        result_path.write_text(RESULT_HEADER, encoding="utf-8")
    row = [
        TEST["test_id"],
        TEST["date"],
        str(TEST["gpu_count"]),
        str(TEST["build_seconds"]),
        str(TEST["eval_seconds"]),
        str(TEST["size_cap_bytes"]),
        parsed["raw_step"],
        parsed["raw_bpb"],
        parsed["post_ema_bpb"],
        parsed["quant_bpb"],
        parsed["sliding_bpb"],
        parsed["ttt_bpb"],
        parsed["total_bytes"],
        parsed["decision"],
        str(log_path),
        str(Path(__file__).resolve()),
    ]
    with result_path.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    preflight_only = "--preflight-only" in sys.argv
    root = repo_root()
    leg = leg_dir()
    logs = leg / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_id = TEST["test_id"] + "_s" + str(MODEL["seed"])
    log_path = logs / (run_id + ".log")
    result_path = leg / "results.tsv"
    config_path = leg / "resolved_config.json"

    env = make_env(root)
    config = resolved_config(root, log_path, result_path)
    print(json.dumps(config, indent=2, sort_keys=True))
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if dry_run:
        print("DRY_RUN: wrote resolved config; not launching")
        return 0
    preflight(root)
    if preflight_only:
        print("PREFLIGHT_OK: not launching")
        return 0
    rc = stream_run(root, env, log_path)
    parsed = parse_result(log_path.read_text(encoding="utf-8", errors="replace"), rc)
    append_result(result_path, log_path, parsed)
    print("RESULT_ROW:", json.dumps(parsed, sort_keys=True))
    print(f"RESULTS: {result_path}")
    print(f"LOG: {log_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
