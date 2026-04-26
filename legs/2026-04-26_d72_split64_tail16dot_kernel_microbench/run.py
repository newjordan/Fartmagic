#!/usr/bin/env python3
"""D72 split64+tail16dot forward kernel microbench against FA3."""

from __future__ import annotations

import json
import hashlib
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


TEST = {
    "test_id": "2026-04-26_d72_split64_tail16dot_kernel_microbench",
    "date": "2026-04-26",
    "hypothesis": "A D72-specific split64+tail16dot forward kernel can remove the 128-wide padded dot tax without the scalar-tail penalty observed in split64+tail8, by handling dims 64..71 through a masked 16-wide tensor-core tail dot while keeping FA3 backward unchanged.",
    "parent_baseline": "Default real-shape whale_fa3_bwd forward path averaged 1.336649x FA3 median; split64+tail8 was correct-ish but rejected at about 2.25x FA3 on its best config.",
    "parent_benchmark_id": "2026-04-26_d72_split64_tail8_kernel_microbench",
    "standard_label": "mechanics_4x_kernel_microbench_scout",
    "gpu_count": 4,
    "build_seconds": 1200,
    "eval_seconds": 1200,
    "size_cap_bytes": 16_000_000,
}

MODEL = {
    "donor_lineage": "whale_l12_dim576_matrixbits5_d72",
    "trainer_contract": "pr1493_decompressed.py",
    "seed": 444,
    "num_layers": 12,
    "model_dim": 576,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 72,
    "xsa_last_n": 11,
    "train_seq_len": 2048,
    "train_microbatch_per_rank": 48,
    "dtype": "bfloat16",
    "tokenizer_path": "./data/tokenizers/fineweb_8192_bpe.model",
    "dataset_dir": "./data/datasets/fineweb10B_sp8192",
}

KERNEL = {
    "attn_mode": "hybrid_microbench",
    "hybrid_backend": "fa3_bwd",
    "benchmark_id": "RUNPY_D72_SPLIT64_TAIL16DOT_KERNEL_MICROBENCH",
    "kernel_source_path": "vault/whale_kernel_triton.py",
    "kernel_source_sha256": "4d8340fc2ed9a656527d25bc8fbd7dfddc691e9ece287ff35017c9b0ee4d26f8",
    "shape": "B=48,T=2048,H=8,KV=4,D=72",
    "causal": 1,
    "correctness_shape": "B=2,T=256,H=8,KV=4,D=72",
    "warmup_iters": 2,
    "timed_iters": 8,
    "profiles": [
        {
            "profile_id": "default_auto_reference",
            "whale_fwd_variant": "default",
            "whale_fwd_d72_config": "",
            "reason": "same-run reference for the current default forward path",
        },
        {
            "profile_id": "tail16dot_128_64_w4_s4",
            "whale_fwd_variant": "d72_split64_tail16dot",
            "whale_fwd_d72_config": "128,64,4,4",
            "reason": "direct successor to rejected tail8 best config, with tensor-core tail math",
        },
        {
            "profile_id": "tail16dot_128_64_w8_s4",
            "whale_fwd_variant": "d72_split64_tail16dot",
            "whale_fwd_d72_config": "128,64,8,4",
            "reason": "same M/N tile with more warps for two-dot pressure",
        },
        {
            "profile_id": "tail16dot_64_64_w4_s4",
            "whale_fwd_variant": "d72_split64_tail16dot",
            "whale_fwd_d72_config": "64,64,4,4",
            "reason": "smaller M tile to reduce live accumulator pressure",
        },
        {
            "profile_id": "tail16dot_128_128_w4_s4",
            "whale_fwd_variant": "d72_split64_tail16dot",
            "whale_fwd_d72_config": "128,128,4,4",
            "reason": "larger N tile to test whether tensor-core tail amortizes loop count",
        },
        {
            "profile_id": "tail16dot_64_128_w4_s4",
            "whale_fwd_variant": "d72_split64_tail16dot",
            "whale_fwd_d72_config": "64,128,4,4",
            "reason": "smaller M plus larger N, checking occupancy versus loop count",
        },
        {
            "profile_id": "tail16dot_128_64_w4_s5",
            "whale_fwd_variant": "d72_split64_tail16dot",
            "whale_fwd_d72_config": "128,64,4,5",
            "reason": "same main tile with one deeper pipeline stage",
        },
        {
            "profile_id": "tail16dot_256_64_w4_s4",
            "whale_fwd_variant": "d72_split64_tail16dot",
            "whale_fwd_d72_config": "256,64,4,4",
            "reason": "larger M tile to test launch/amortization at real T=2048",
        },
    ],
}

RESULT_HEADER = (
    "test_id\tdate\tgpu_count\tbuild_seconds\teval_seconds\tsize_cap_bytes\t"
    "raw_step\traw_bpb\tpost_ema_bpb\tquant_bpb\tsliding_bpb\t"
    "ttt_bpb\ttotal_bytes\tdecision\tlog_path\trun_py\n"
)

MICRO_HEADER = (
    "profile_id\tgpu\tmode\tmedian_ms\tmin_ms\tp90_ms\tmean_ms\t"
    "ratio_vs_fa3_median\tout_max_abs\tdq_max_abs\tdk_max_abs\tdv_max_abs\terror\n"
)

CHILD_CODE = r'''
import json
import os
import time
import traceback

cfg = json.loads(os.environ["BENCH_CONFIG_JSON"])
root = cfg["root"]
os.chdir(root)

import torch
from flash_attn_interface import flash_attn_func
from vault.whale_kernel_triton import whale_fwd_fa3_bwd

torch.manual_seed(int(cfg["seed"]) + int(cfg["gpu"]))
torch.cuda.set_device(0)
device = torch.device("cuda")
dtype = torch.bfloat16

B = int(cfg["B"])
T = int(cfg["T"])
H = int(cfg["H"])
KV = int(cfg["KV"])
D = int(cfg["D"])
warmup = int(cfg["warmup_iters"])
timed = int(cfg["timed_iters"])

q0 = torch.randn(B, T, H, D, device=device, dtype=dtype)
k0 = torch.randn(B, T, KV, D, device=device, dtype=dtype)
v0 = torch.randn(B, T, KV, D, device=device, dtype=dtype)


def whale_call(q, k, v, variant, d72_config):
    old_variant = os.environ.get("WHALE_FWD_VARIANT")
    old_d72_config = os.environ.get("WHALE_FWD_D72_CONFIG")
    try:
        os.environ["WHALE_FWD_VARIANT"] = variant
        if d72_config:
            os.environ["WHALE_FWD_D72_CONFIG"] = d72_config
        else:
            os.environ.pop("WHALE_FWD_D72_CONFIG", None)
        return whale_fwd_fa3_bwd(q, k, v, causal=True)
    finally:
        if old_variant is None:
            os.environ.pop("WHALE_FWD_VARIANT", None)
        else:
            os.environ["WHALE_FWD_VARIANT"] = old_variant
        if old_d72_config is None:
            os.environ.pop("WHALE_FWD_D72_CONFIG", None)
        else:
            os.environ["WHALE_FWD_D72_CONFIG"] = old_d72_config


def stats(times):
    times = sorted(times)
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "p90_ms": times[min(len(times) - 1, int(len(times) * 0.9))],
        "mean_ms": sum(times) / len(times),
    }


def bench(fn):
    for _ in range(warmup):
        q = q0.clone().requires_grad_(True)
        k = k0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        out = fn(q, k, v)
        out.sum().backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        q = q0.clone().requires_grad_(True)
        k = k0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(q, k, v)
        out.sum().backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return stats(times)


def whale_fn(q, k, v):
    return whale_call(q, k, v, cfg["variant"], cfg["d72_config"])


def fa3_fn(q, k, v):
    return flash_attn_func(q, k, v, causal=True)


result = {
    "profile_id": cfg["profile_id"],
    "gpu": cfg["gpu"],
    "shape": {"B": B, "T": T, "H": H, "KV": KV, "D": D},
    "env": {
        "WHALE_FWD_VARIANT": os.environ.get("WHALE_FWD_VARIANT", ""),
        "WHALE_FWD_D72_CONFIG": os.environ.get("WHALE_FWD_D72_CONFIG", ""),
        "WHALE_BWD_VARIANT": os.environ.get("WHALE_BWD_VARIANT", ""),
    },
}


def correctness():
    torch.manual_seed(10_000 + int(cfg["gpu"]))
    b_small, t_small = 2, 256
    q_ref = torch.randn(b_small, t_small, H, D, device=device, dtype=dtype, requires_grad=True)
    k_ref = torch.randn(b_small, t_small, KV, D, device=device, dtype=dtype, requires_grad=True)
    v_ref = torch.randn(b_small, t_small, KV, D, device=device, dtype=dtype, requires_grad=True)
    q_new = q_ref.detach().clone().requires_grad_(True)
    k_new = k_ref.detach().clone().requires_grad_(True)
    v_new = v_ref.detach().clone().requires_grad_(True)
    dout = torch.randn(b_small, t_small, H, D, device=device, dtype=dtype)

    out_ref = whale_call(q_ref, k_ref, v_ref, "default", "")
    out_new = whale_call(q_new, k_new, v_new, cfg["variant"], cfg["d72_config"])
    (out_ref * dout).sum().backward()
    (out_new * dout).sum().backward()
    torch.cuda.synchronize()
    return {
        "out_max_abs": float((out_new - out_ref).abs().max().item()),
        "out_mean_abs": float((out_new - out_ref).abs().mean().item()),
        "dq_max_abs": float((q_new.grad - q_ref.grad).abs().max().item()),
        "dk_max_abs": float((k_new.grad - k_ref.grad).abs().max().item()),
        "dv_max_abs": float((v_new.grad - v_ref.grad).abs().max().item()),
    }


try:
    result["correctness_vs_default"] = correctness()
except Exception as exc:
    result["correctness_error"] = repr(exc)
    result["correctness_traceback"] = traceback.format_exc()

try:
    result["whale_fa3_bwd"] = bench(whale_fn)
except Exception as exc:
    result["whale_error"] = repr(exc)
    result["whale_traceback"] = traceback.format_exc()

try:
    result["fa3"] = bench(fa3_fn)
except Exception as exc:
    result["fa3_error"] = repr(exc)
    result["fa3_traceback"] = traceback.format_exc()

if "whale_fa3_bwd" in result and "fa3" in result:
    result["ratio_vs_fa3_median"] = (
        result["whale_fa3_bwd"]["median_ms"] / result["fa3"]["median_ms"]
    )

print("JSON_RESULT " + json.dumps(result, sort_keys=True), flush=True)
'''


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def leg_dir() -> Path:
    return Path(__file__).resolve().parent


def resolved_config(root: Path, log_path: Path, result_path: Path, micro_path: Path) -> dict[str, object]:
    return {
        "test": TEST,
        "model": MODEL,
        "kernel": KERNEL,
        "repo_root": str(root),
        "run_py": str(Path(__file__).resolve()),
        "log_path": str(log_path),
        "result_path": str(result_path),
        "microbench_path": str(micro_path),
    }


def preflight(root: Path) -> None:
    required = [
        root / "vault/whale_kernel_triton.py",
        root / MODEL["trainer_contract"],
        root / MODEL["tokenizer_path"],
        root / MODEL["dataset_dir"],
        root / "scripts/pod_stack_guard.sh",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("FATAL: missing required paths:\n" + "\n".join(missing))
    kernel_path = root / KERNEL["kernel_source_path"]
    kernel_sha = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    if kernel_sha != KERNEL["kernel_source_sha256"]:
        raise SystemExit(
            "FATAL: kernel source hash mismatch "
            f"expected={KERNEL['kernel_source_sha256']} actual={kernel_sha} path={kernel_path}"
        )
    if not Path("/venv/main/bin/python3").is_file():
        raise SystemExit("FATAL: /venv/main/bin/python3 not found")
    guard = subprocess.run(
        ["bash", "scripts/pod_stack_guard.sh", "verify"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if guard.returncode != 0:
        raise SystemExit("FATAL: pod stack guard failed:\n" + guard.stdout + guard.stderr)


def child_env(root: Path, profile: dict[str, str], gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = "/venv/main/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["WHALE_BWD_VARIANT"] = "fused_delta_tma"
    env["WHALE_FUSED_DELTA_T_MAX"] = "3072"
    if profile["whale_fwd_variant"]:
        env["WHALE_FWD_VARIANT"] = profile["whale_fwd_variant"]
    else:
        env.pop("WHALE_FWD_VARIANT", None)
    if profile["whale_fwd_d72_config"]:
        env["WHALE_FWD_D72_CONFIG"] = profile["whale_fwd_d72_config"]
    else:
        env.pop("WHALE_FWD_D72_CONFIG", None)
    env["BENCH_CONFIG_JSON"] = json.dumps(
        {
            "root": str(root),
            "seed": MODEL["seed"],
            "gpu": gpu,
            "profile_id": profile["profile_id"],
            "variant": profile["whale_fwd_variant"],
            "d72_config": profile["whale_fwd_d72_config"],
            "B": MODEL["train_microbatch_per_rank"],
            "T": MODEL["train_seq_len"],
            "H": MODEL["num_heads"],
            "KV": MODEL["num_kv_heads"],
            "D": MODEL["head_dim"],
            "warmup_iters": KERNEL["warmup_iters"],
            "timed_iters": KERNEL["timed_iters"],
        },
        sort_keys=True,
    )
    return env


def launch_profile_batch(
    root: Path,
    profiles: list[dict[str, str]],
    logs: Path,
    batch_index: int,
) -> list[dict[str, object]]:
    procs: list[tuple[int, dict[str, str], subprocess.Popen[str], Path]] = []
    for gpu, profile in enumerate(profiles):
        worker_log = logs / f"batch{batch_index}_{profile['profile_id']}_gpu{gpu}.log"
        env = child_env(root, profile, gpu)
        proc = subprocess.Popen(
            ["/venv/main/bin/python3", "-c", CHILD_CODE],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        procs.append((gpu, profile, proc, worker_log))

    deadline = time.time() + 900
    results: list[dict[str, object]] = []
    for gpu, profile, proc, worker_log in procs:
        assert proc.stdout is not None
        with worker_log.open("w", encoding="utf-8") as log:
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            try:
                while True:
                    if time.time() > deadline:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        result = {
                            "profile_id": profile["profile_id"],
                            "gpu": gpu,
                            "error": "timeout",
                        }
                        results.append(result)
                        break
                    events = selector.select(timeout=1.0)
                    for key, _ in events:
                        line = key.fileobj.readline()
                        if line:
                            sys.stdout.write(line)
                            log.write(line)
                            log.flush()
                            if line.startswith("JSON_RESULT "):
                                results.append(json.loads(line[len("JSON_RESULT "):]))
                    if proc.poll() is not None:
                        tail = proc.stdout.read()
                        if tail:
                            sys.stdout.write(tail)
                            log.write(tail)
                            for line in tail.splitlines():
                                if line.startswith("JSON_RESULT "):
                                    results.append(json.loads(line[len("JSON_RESULT "):]))
                        if proc.returncode != 0 and not any(
                            r.get("profile_id") == profile["profile_id"] and r.get("gpu") == gpu
                            for r in results
                        ):
                            results.append(
                                {
                                    "profile_id": profile["profile_id"],
                                    "gpu": gpu,
                                    "error": f"rc_{proc.returncode}",
                                }
                            )
                        break
            finally:
                selector.unregister(proc.stdout)
                selector.close()
                proc.stdout.close()
    return results


def append_result(result_path: Path, log_path: Path, decision: str) -> None:
    if not result_path.exists():
        result_path.write_text(RESULT_HEADER, encoding="utf-8")
    row = [
        TEST["test_id"],
        TEST["date"],
        str(TEST["gpu_count"]),
        str(TEST["build_seconds"]),
        str(TEST["eval_seconds"]),
        str(TEST["size_cap_bytes"]),
        "NA",
        "NA",
        "NA",
        "NA",
        "NA",
        "NA",
        "NA",
        decision,
        str(log_path),
        str(Path(__file__).resolve()),
    ]
    with result_path.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def write_microbench_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(MICRO_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            profile_id = str(row.get("profile_id", "NA"))
            gpu = str(row.get("gpu", "NA"))
            ratio = row.get("ratio_vs_fa3_median", "NA")
            correctness = row.get("correctness_vs_default")
            if isinstance(correctness, dict):
                out_max = f"{float(correctness.get('out_max_abs', float('nan'))):.6f}"
                dq_max = f"{float(correctness.get('dq_max_abs', float('nan'))):.6f}"
                dk_max = f"{float(correctness.get('dk_max_abs', float('nan'))):.6f}"
                dv_max = f"{float(correctness.get('dv_max_abs', float('nan'))):.6f}"
            else:
                out_max = dq_max = dk_max = dv_max = "NA"
            for mode in ("whale_fa3_bwd", "fa3"):
                stats = row.get(mode)
                error = (
                    row.get("error")
                    or row.get("correctness_error")
                    or row.get("whale_error" if mode == "whale_fa3_bwd" else "fa3_error")
                    or ""
                )
                if isinstance(stats, dict):
                    values = [
                        profile_id,
                        gpu,
                        mode,
                        f"{float(stats['median_ms']):.6f}",
                        f"{float(stats['min_ms']):.6f}",
                        f"{float(stats['p90_ms']):.6f}",
                        f"{float(stats['mean_ms']):.6f}",
                        f"{float(ratio):.6f}" if isinstance(ratio, (int, float)) else "NA",
                        out_max,
                        dq_max,
                        dk_max,
                        dv_max,
                        str(error),
                    ]
                else:
                    values = [
                        profile_id,
                        gpu,
                        mode,
                        "NA",
                        "NA",
                        "NA",
                        "NA",
                        "NA",
                        out_max,
                        dq_max,
                        dk_max,
                        dv_max,
                        str(error),
                    ]
                f.write("\t".join(values) + "\n")


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
    micro_path = leg / "microbench.tsv"
    json_path = leg / "microbench.json"
    config_path = leg / "resolved_config.json"

    config = resolved_config(root, log_path, result_path, micro_path)
    print(json.dumps(config, indent=2, sort_keys=True))
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        print("DRY_RUN: wrote resolved config; not launching")
        return 0
    preflight(root)
    if preflight_only:
        print("PREFLIGHT_OK: not launching")
        return 0

    all_rows: list[dict[str, object]] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
        log.flush()
        profiles = list(KERNEL["profiles"])
        for batch_start in range(0, len(profiles), TEST["gpu_count"]):
            batch_index = batch_start // TEST["gpu_count"]
            batch = profiles[batch_start:batch_start + TEST["gpu_count"]]
            batch_ids = ",".join(profile["profile_id"] for profile in batch)
            msg = f"BATCH_START {batch_index} {batch_ids}\n"
            print(msg, end="")
            log.write(msg)
            log.flush()
            rows = launch_profile_batch(root, batch, logs, batch_index)
            all_rows.extend(rows)
            for row in rows:
                line = "PROFILE_RESULT " + json.dumps(row, sort_keys=True) + "\n"
                print(line, end="")
                log.write(line)
                log.flush()

    write_microbench_rows(micro_path, all_rows)
    json_path.write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision = "kernel_microbench_complete"
    if any(
        "error" in row or "correctness_error" in row or "whale_error" in row or "fa3_error" in row
        for row in all_rows
    ):
        decision = "kernel_microbench_partial"
    append_result(result_path, log_path, decision)
    print(f"MICROBENCH_TSV: {micro_path}")
    print(f"MICROBENCH_JSON: {json_path}")
    print(f"RESULTS: {result_path}")
    print(f"LOG: {log_path}")
    return 0 if decision == "kernel_microbench_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
