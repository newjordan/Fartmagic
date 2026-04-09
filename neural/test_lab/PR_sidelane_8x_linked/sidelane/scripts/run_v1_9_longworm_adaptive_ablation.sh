#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="${REPO_ROOT}"
elif git -C "${ROOT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "${ROOT_DIR}" rev-parse --show-toplevel)"
else
  REPO_ROOT="$(cd -- "${ROOT_DIR}/../../../../.." && pwd)"
fi
PG_LAB_ROOT="${PG_LAB_ROOT:-${REPO_ROOT}/neural}"

if [[ -z "${TRAIN_PY:-}" ]]; then
  TRAIN_PY="${PG_LAB_ROOT}/experiments/Longworm/train_longworm.py"
fi

if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  for candidate in \
    "${PG_LAB_ROOT}/data/tokenizers/fineweb_1024_bpe.model" \
    "${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model" \
    "${REPO_ROOT}/data/fineweb_1024_bpe.model"; do
    if [[ -f "${candidate}" ]]; then
      TOKENIZER_PATH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ -f "${REPO_ROOT}/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" && -f "${REPO_ROOT}/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]]; then
    DATA_PATH="${REPO_ROOT}/data/datasets/fineweb10B_sp1024"
  else
    DATA_PATH="${PG_LAB_ROOT}/data/datasets/fineweb10B_sp1024"
  fi
fi

if [[ -z "${TRAIN_PY:-}" || ! -f "${TRAIN_PY}" ]]; then
  echo "TRAIN_PY not found. Set TRAIN_PY=/abs/path/to/experiments/Longworm/train_longworm.py" >&2
  exit 1
fi
if [[ -z "${TOKENIZER_PATH:-}" || ! -f "${TOKENIZER_PATH}" ]]; then
  echo "TOKENIZER_PATH not found. Set TOKENIZER_PATH=/abs/path/to/fineweb_1024_bpe.model" >&2
  exit 1
fi
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "DATA_PATH not found: ${DATA_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  if ! python3 - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "CUDA not available in this shell. Run on GPU or set SKIP_CUDA_CHECK=1." >&2
    exit 1
  fi
fi

if ! python3 - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("brotli") else 1)
PY
then
  if [[ "${AUTO_INSTALL_BROTLI:-1}" == "1" ]]; then
    echo "brotli_missing: installing brotli into current python env..."
    python3 -m pip install --no-cache-dir brotli
  else
    echo "brotli_missing: set AUTO_INSTALL_BROTLI=1 or install manually: python3 -m pip install brotli" >&2
    exit 1
  fi
fi

TORCHRUN=(python3 -m torch.distributed.run)
SEED="${SEED:-445}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
PROJECT_CODENAME="${PROJECT_CODENAME:-longworm_v1_9}"
RUNS_SUBDIR="${RUNS_SUBDIR:-${PROJECT_CODENAME}_adaptive_ablation_brotli}"
SUBMISSION_PROFILE="${SUBMISSION_PROFILE:-track_10min_16mb}"
OUT_DIR="${ROOT_DIR}/runs/${RUNS_SUBDIR}/${TS}_s${SEED}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
LEADERBOARD="${OUT_DIR}/leaderboard.tsv"

export DATA_PATH TOKENIZER_PATH SEED
case "${SUBMISSION_PROFILE}" in
  track_10min_16mb)
    export ITERATIONS="${ITERATIONS:-16000}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-590}"
    ;;
  longform)
    export ITERATIONS="${ITERATIONS:-20000}"
    export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-2400}"
    ;;
  *)
    echo "Unsupported SUBMISSION_PROFILE=${SUBMISSION_PROFILE}. Use track_10min_16mb or longform." >&2
    exit 1
    ;;
esac

if [[ "${SUBMISSION_PROFILE}" == "track_10min_16mb" ]]; then
  TRACK_4K_BPB="${TRACK_4K_BPB:-0}"
else
  TRACK_4K_BPB="${TRACK_4K_BPB:-1}"
fi
TRACK_EVAL_SEQ_LEN="${TRACK_EVAL_SEQ_LEN:-4096}"
TRACK_EVAL_STRIDE="${TRACK_EVAL_STRIDE:-0}"
LEADERBOARD_METRIC_PREF="${LEADERBOARD_METRIC_PREF:-submission}" # submission|tracked_4k

export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export COMPILE_ENABLED="${COMPILE_ENABLED:-0}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"
export TORCHDYNAMO_OPTIMIZE_DDP="${TORCHDYNAMO_OPTIMIZE_DDP:-0}"
export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
export SKIP_GPTQ="${SKIP_GPTQ:-1}"
export SKIP_EMA="${SKIP_EMA:-1}"
export SWA_ENABLED="${SWA_ENABLED:-0}"

ARMS=(
  "39_v1_9_longworm_ssm_ttt_baseline_l8_d480_h12_kv4_non_ngram_brotli|control"
  "40_v1_9_longworm_loss_gated_ttt_l8_d480_h12_kv4_non_ngram_brotli|candidate"
  "41_v1_9_longworm_loss_drift_gated_ttt_l8_d480_h12_kv4_non_ngram_brotli|candidate"
)

printf "lane\tarm\trole\tstatus\tcompressor\tmodel_params\tdiag_bpb\tsw_bpb\tsw_bpb_4k\ttracked_eval_seq_len\ttotal_size_mixed_bytes\tstep_avg_ms\tsteps_done\tlog\n" > "${SUMMARY}"

echo "============================================================"
echo "Project: ${PROJECT_CODENAME}"
echo "Longworm v1.9 adaptive ablation (SSM + TTT + loss/drift-gated test-time compute)"
echo "out_dir: ${OUT_DIR}"
echo "train_py: ${TRAIN_PY}"
echo "data_path: ${DATA_PATH}"
echo "tokenizer_path: ${TOKENIZER_PATH}"
echo "seed: ${SEED}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "submission_profile: ${SUBMISSION_PROFILE}"
echo "iterations: ${ITERATIONS}"
echo "max_wallclock_seconds: ${MAX_WALLCLOCK_SECONDS}"
echo "track_4k_bpb: ${TRACK_4K_BPB}"
echo "leaderboard_metric_pref: ${LEADERBOARD_METRIC_PREF}"
if [[ "${TRACK_4K_BPB}" == "1" ]]; then
  echo "track_eval_seq_len: ${TRACK_EVAL_SEQ_LEN}"
  echo "track_eval_stride: ${TRACK_EVAL_STRIDE}"
fi
if [[ -n "${ARM_ONLY:-}" ]]; then
  echo "arm_only: ${ARM_ONLY}"
fi
echo "============================================================"

for arm_spec in "${ARMS[@]}"; do
  IFS="|" read -r arm_name arm_role <<< "${arm_spec}"
  if [[ -n "${ARM_ONLY:-}" && "${arm_name}" != "${ARM_ONLY}" ]]; then
    continue
  fi

  arm_file="${ROOT_DIR}/04_ssm_e2e_ttt_long_context/arms/${arm_name}.env"
  if [[ ! -f "${arm_file}" ]]; then
    echo "Missing arm file: ${arm_file}" >&2
    exit 1
  fi

  log_path="${OUT_DIR}/${arm_name}.log"
  echo
  echo "--- 04_ssm_e2e_ttt_long_context / ${arm_name} (${arm_role}) ---"

  tracked_eval_seq_len="-"
  if [[ "${TRACK_4K_BPB}" == "1" ]]; then
    tracked_eval_seq_len="${TRACK_EVAL_SEQ_LEN}"
  fi

  arm_status="ok"
  if (
    set -a
    # shellcheck disable=SC1090
    source "${arm_file}"
    set +a
    if [[ "${TRACK_4K_BPB}" == "1" ]]; then
      export EVAL_SEQ_LEN="${TRACK_EVAL_SEQ_LEN}"
      if [[ "${TRACK_EVAL_STRIDE}" != "0" ]]; then
        export EVAL_STRIDE="${TRACK_EVAL_STRIDE}"
      fi
    fi
    export RUN_ID="${PROJECT_CODENAME}_04_longctx_${arm_name}_s${SEED}_${TS}"
    "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_PY}"
  ) 2>&1 | tee "${log_path}"; then
    arm_status="ok"
  else
    arm_status="fail"
    echo "arm_failed:04_ssm_e2e_ttt_long_context/${arm_name} (continuing sweep)" >&2
  fi

  if ! grep -q 'Serialized model mixed+brotli:' "${log_path}"; then
    echo "compression_check_failed:${arm_name} expected brotli artifact path not detected" >&2
    arm_status="fail_compressor"
  fi

  model_params="$(grep -oP 'model_params:\K[0-9]+' "${log_path}" | tail -1 || true)"
  diag_bpb="$(grep -oP 'DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log_path}" | tail -1 || true)"
  sw_bpb="$(grep -oP '(?:final_int8_zlib_roundtrip_exact|final_sliding_window(?:_s[0-9]+)?_exact) val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log_path}" | tail -1 || true)"
  compressor="$(grep -oP 'Serialized model mixed\+\K[a-z0-9_]+' "${log_path}" | tail -1 || true)"
  size_mixed="$(grep -oP 'Total submission size mixed\+\w+: \K[0-9]+' "${log_path}" | tail -1 || true)"
  step_avg_ms="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+ train_time:[0-9]+ms step_avg:\K[0-9.]+' "${log_path}" | tail -1 || true)"
  steps_done="$(grep -oP 'stopping_early: wallclock_cap train_time:[0-9]+ms step:\K[0-9]+' "${log_path}" | tail -1 || true)"
  if [[ -z "${steps_done}" ]]; then
    steps_done="$(grep -oP 'step:\K[0-9]+(?=/[0-9]+ val_loss:)' "${log_path}" | tail -1 || true)"
  fi

  [[ -z "${model_params}" ]] && model_params="-"
  [[ -z "${diag_bpb}" ]] && diag_bpb="-"
  [[ -z "${sw_bpb}" ]] && sw_bpb="-"
  sw_bpb_4k="-"
  if [[ "${TRACK_4K_BPB}" == "1" && "${TRACK_EVAL_SEQ_LEN}" -ge 4096 ]]; then
    sw_bpb_4k="${sw_bpb}"
  fi
  [[ -z "${compressor}" ]] && compressor="-"
  [[ -z "${size_mixed}" ]] && size_mixed="-"
  [[ -z "${step_avg_ms}" ]] && step_avg_ms="-"
  [[ -z "${steps_done}" ]] && steps_done="-"

  printf "%s_v1_9_adaptive\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${PROJECT_CODENAME}" "${arm_name}" "${arm_role}" "${arm_status}" "${compressor}" "${model_params}" "${diag_bpb}" "${sw_bpb}" "${sw_bpb_4k}" "${tracked_eval_seq_len}" "${size_mixed}" "${step_avg_ms}" "${steps_done}" "${log_path}" \
    >> "${SUMMARY}"
done

python3 - "${SUMMARY}" "${LEADERBOARD}" "${LEADERBOARD_METRIC_PREF}" <<'PY'
import csv
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
leaderboard_path = Path(sys.argv[2])
metric_pref = (sys.argv[3] if len(sys.argv) > 3 else "submission").strip().lower()

with summary_path.open() as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

def parse(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.inf

ok_rows = [r for r in rows if r.get("status") == "ok"]
for r in ok_rows:
    tracked = r.get("sw_bpb_4k", "")
    base = r.get("sw_bpb", "")
    if metric_pref == "tracked_4k":
        metric_src = tracked if tracked not in ("", "-") else base
    else:
        metric_src = base if base not in ("", "-") else tracked
    r["_metric"] = parse(metric_src)

ok_rows = [r for r in ok_rows if math.isfinite(r.get("_metric", math.inf))]
ok_rows.sort(key=lambda r: r["_metric"])

with leaderboard_path.open("w") as f:
    f.write("rank\tarm\trole\tmetric_bpb\tsw_bpb\tsw_bpb_4k\ttracked_eval_seq_len\ttotal_size_mixed_bytes\tstep_avg_ms\tsteps_done\tlog\n")
    for idx, r in enumerate(ok_rows, start=1):
        f.write(
            f"{idx}\t{r['arm']}\t{r['role']}\t{r['_metric']:.8f}\t{r['sw_bpb']}\t{r['sw_bpb_4k']}\t{r['tracked_eval_seq_len']}\t{r['total_size_mixed_bytes']}\t{r['step_avg_ms']}\t{r['steps_done']}\t{r['log']}\n"
        )

print(f"summary={summary_path}")
print(f"leaderboard={leaderboard_path}")
if ok_rows:
    best = ok_rows[0]
    print(f"best={best['arm']} metric_bpb={best['_metric']:.8f}")
else:
    print("best=unavailable")
PY

echo
echo "${PROJECT_CODENAME}_v1_9_adaptive_ablation_complete"
echo "summary: ${SUMMARY}"
echo "leaderboard: ${LEADERBOARD}"
