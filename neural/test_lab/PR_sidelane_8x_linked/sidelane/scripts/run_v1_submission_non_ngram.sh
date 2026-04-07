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
  for candidate in \
    "${PG_LAB_ROOT}/experiments/rascal_hunt_2k/train_gpt.py" \
    "${PG_LAB_ROOT}/experiments/Rascal_III/train_gpt.py" \
    "${REPO_ROOT}/train_gpt.py"; do
    if [[ -f "${candidate}" ]]; then
      TRAIN_PY="${candidate}"
      break
    fi
  done
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
  echo "TRAIN_PY not found. Set TRAIN_PY=/abs/path/to/train_gpt.py" >&2
  exit 1
fi
if [[ -z "${TOKENIZER_PATH:-}" || ! -f "${TOKENIZER_PATH}" ]]; then
  echo "TOKENIZER_PATH not found. Set TOKENIZER_PATH=/abs/path/to/fineweb_1024_bpe.model" >&2
  exit 1
fi
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "DATA_PATH not found: ${DATA_PATH}" >&2
  echo "Set DATA_PATH=/abs/path/to/tokenized_dataset_dir" >&2
  exit 1
fi

if [[ "${SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  if ! python3 - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "CUDA not available in this shell. Run this on H100/A100 (or set SKIP_CUDA_CHECK=1)." >&2
    exit 1
  fi
fi

ARM_FILE="${ROOT_DIR}/04_ssm_e2e_ttt_long_context/arms/21_v1_longctx_ttt_densehash_non_ngram.env"
if [[ ! -f "${ARM_FILE}" ]]; then
  echo "Missing arm file: ${ARM_FILE}" >&2
  exit 1
fi

TORCHRUN=(python3 -m torch.distributed.run)
SEED="${SEED:-445}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TS="$(date +%Y%m%d_%H%M%S)"
RUNS_SUBDIR="${RUNS_SUBDIR:-v1_submission_non_ngram}"
OUT_DIR="${ROOT_DIR}/runs/${RUNS_SUBDIR}/${TS}_s${SEED}"
mkdir -p "${OUT_DIR}"
LOG_PATH="${OUT_DIR}/v1_longctx_ttt_densehash_non_ngram.log"
SUMMARY="${OUT_DIR}/summary.tsv"

export DATA_PATH TOKENIZER_PATH SEED
export ITERATIONS="${ITERATIONS:-800}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-2400}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export COMPILE_ENABLED="${COMPILE_ENABLED:-0}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"
export TORCHDYNAMO_OPTIMIZE_DDP="${TORCHDYNAMO_OPTIMIZE_DDP:-0}"
export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
export SKIP_GPTQ="${SKIP_GPTQ:-1}"
export SKIP_EMA="${SKIP_EMA:-1}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MODEL_DIM="${MODEL_DIM:-384}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export NUM_HEADS="${NUM_HEADS:-6}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-3}"
export MLP_MULT="${MLP_MULT:-3.0}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export VE_ENABLED="${VE_ENABLED:-1}"
export VE_DIM="${VE_DIM:-128}"
export VE_LAYERS="${VE_LAYERS:-7,8}"
export DTG_ENABLED="${DTG_ENABLED:-0}"
export USE_CRAWLER="${USE_CRAWLER:-0}"
export NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-9}"
export NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-1}"
export CRAWLER_LOOPS="${CRAWLER_LOOPS:-3}"
export CRAWLER_MLP_MULT="${CRAWLER_MLP_MULT:-6.0}"
export INST_DIM="${INST_DIM:-32}"
export CRAWLER_LOOP_ROPE_SCALES="${CRAWLER_LOOP_ROPE_SCALES:-9,1,1}"
export CRAWLER_LOOP_SMEAR="${CRAWLER_LOOP_SMEAR:-0}"

echo "============================================================"
echo "V1 submission test (non-ngram)"
echo "arm_file: ${ARM_FILE}"
echo "out_dir: ${OUT_DIR}"
echo "train_py: ${TRAIN_PY}"
echo "data_path: ${DATA_PATH}"
echo "tokenizer_path: ${TOKENIZER_PATH}"
echo "seed: ${SEED}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "iterations: ${ITERATIONS}"
echo "max_wallclock_seconds: ${MAX_WALLCLOCK_SECONDS}"
echo "============================================================"

if (
  set -a
  # shellcheck disable=SC1090
  source "${ARM_FILE}"
  set +a
  # Hard legal gate: keep external n-gram eval disabled.
  export NGRAM_EVAL_ORDER=0
  export NGRAM_EVAL_ALPHA=0.0
  export NGRAM_EVAL_ADAPTIVE=0
  export CUBRIC_CADENCE=0
  export RUN_ID="v1_non_ngram_04_longctx_ttt_densehash_s${SEED}_${TS}"
  "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_PY}"
) 2>&1 | tee "${LOG_PATH}"; then
  status="ok"
else
  status="fail"
fi

if grep -qE 'final_sliding_window_ngram[0-9]+' "${LOG_PATH}"; then
  echo "illegal_metric_detected: n-gram evaluation outputs found in log" >&2
  status="fail_legal"
fi

model_params="$(grep -oP 'model_params:\K[0-9]+' "${LOG_PATH}" | tail -1 || true)"
diag_bpb="$(grep -oP 'DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG_PATH}" | tail -1 || true)"
sw_bpb="$(grep -oP '(?:final_int8_zlib_roundtrip_exact|final_sliding_window(?:_s[0-9]+)?_exact) val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG_PATH}" | tail -1 || true)"
size_mixed="$(grep -oP 'Total submission size mixed\+zlib: \K[0-9]+' "${LOG_PATH}" | tail -1 || true)"
size_int6="$(grep -oP 'Total submission size int6\+\w+: \K[0-9]+' "${LOG_PATH}" | tail -1 || true)"
step_avg_ms="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+ train_time:[0-9]+ms step_avg:\K[0-9.]+' "${LOG_PATH}" | tail -1 || true)"

[[ -z "${model_params}" ]] && model_params="-"
[[ -z "${diag_bpb}" ]] && diag_bpb="-"
[[ -z "${sw_bpb}" ]] && sw_bpb="-"
[[ -z "${size_mixed}" ]] && size_mixed="-"
[[ -z "${size_int6}" ]] && size_int6="-"
[[ -z "${step_avg_ms}" ]] && step_avg_ms="-"

printf "lane\tarm\tstatus\tmodel_params\tdiag_bpb\tsw_bpb\ttotal_size_mixed_zlib_bytes\ttotal_size_int6_bytes\tstep_avg_ms\tlog\n" > "${SUMMARY}"
printf "v1_non_ngram\t21_v1_longctx_ttt_densehash_non_ngram\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "${status}" "${model_params}" "${diag_bpb}" "${sw_bpb}" "${size_mixed}" "${size_int6}" "${step_avg_ms}" "${LOG_PATH}" \
  >> "${SUMMARY}"

echo
echo "v1_submission_test_complete"
echo "summary: ${SUMMARY}"
echo "log: ${LOG_PATH}"
echo "status: ${status}"
