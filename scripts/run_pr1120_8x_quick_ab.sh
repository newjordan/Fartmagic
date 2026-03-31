#!/usr/bin/env bash
set -euo pipefail

# Quick 8xH100 A/B launcher for Rascal GPTQ stream vs insta-cache.
# Usage:
#   bash scripts/run_pr1120_8x_quick_ab.sh
# Optional overrides:
#   SEED=444 NPROC_PER_NODE=8 GPTQ_RESERVE_MS=9000 GPTQ_CALIB_SAMPLES=256 bash scripts/run_pr1120_8x_quick_ab.sh

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

COPY_DIR="analysis/pr1120_racecar_lab/copies"
RUN_DIR="analysis/pr1120_racecar_lab/runs_8x_econ"
TRAIN_COPY="${COPY_DIR}/train_gpt_rascal_sota_local.py"
mkdir -p "${COPY_DIR}" "${RUN_DIR}"

# Bootstrap trainer copy if needed.
if [ ! -f "${TRAIN_COPY}" ]; then
  for src in \
    "scripts/train_gpt_rascal_insta_cache.py" \
    "experiments/Rascal_Master/train_gpt.py" \
    "experiments/SOTA/2026-03-30_JUNKYARD_RAT_RASCAL_II_nogptq/train_gpt.py" \
    "records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py"
  do
    if [ -f "${src}" ]; then
      cp -f "${src}" "${TRAIN_COPY}"
      echo "[bootstrap] copied ${src} -> ${TRAIN_COPY}"
      break
    fi
  done
fi

if [ ! -f "${TRAIN_COPY}" ]; then
  echo "FATAL: could not locate a Rascal trainer source to copy"
  exit 1
fi

: "${PYTHON_BIN:=python3}"
: "${NPROC_PER_NODE:=8}"
: "${SEED:=444}"
: "${MAX_WALLCLOCK_SECONDS:=600}"
: "${GPTQ_RESERVE_MS:=9000}"
: "${GPTQ_CALIB_SAMPLES:=256}"
: "${GPTQ_CACHE_SEQS_PER_STEP:=1}"

echo "[preflight] torch/cuda/gpu:"
"${PYTHON_BIN}" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"

if "${PYTHON_BIN}" -c "from flash_attn_interface import flash_attn_func; print('FA3_OK')" >/tmp/pr1120_fa3_check.txt 2>&1; then
  echo "[preflight] $(cat /tmp/pr1120_fa3_check.txt)"
else
  echo "[preflight] WARNING: flash_attn_interface import failed (likely no FA3)."
  echo "[preflight] Detail:"
  sed -n '1,3p' /tmp/pr1120_fa3_check.txt || true
fi

COMMON_ENV=(
  "SEED=${SEED}"
  "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}"
  "SKIP_GPTQ=0"
  "GPTQ_RESERVE_MS=${GPTQ_RESERVE_MS}"
  "GPTQ_CALIB_SAMPLES=${GPTQ_CALIB_SAMPLES}"
  "LOADER_MODE=coprime"
  "COPRIME_MAX_LOADED_SHARDS=1"
  "COPRIME_SHARDS_PER_BATCH=1"
  "COPRIME_SHARD_HOLD_STEPS=64"
  "XSA_LAST_N=11"
  "BIGRAM_VOCAB_SIZE=2048"
  "BIGRAM_DIM=128"
  "ROPE_DIMS=16"
  "SWA_EVERY=50"
  "NGRAM_EVAL_ORDER=0"
  "MTP_NUM_HEADS=0"
)

run_case() {
  local name="$1"
  shift
  local log="${RUN_DIR}/${name}.log"
  echo "[run] ${name}"
  env "${COMMON_ENV[@]}" "$@" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_COPY}" \
    2>&1 | tee "${log}"
}

if grep -q "GPTQ_INSTA_CACHE" "${TRAIN_COPY}"; then
  run_case "stream_seed${SEED}" "GPTQ_INSTA_CACHE=0"
  run_case "insta_seed${SEED}" "GPTQ_INSTA_CACHE=1" "GPTQ_CACHE_SEQS_PER_STEP=${GPTQ_CACHE_SEQS_PER_STEP}"
else
  echo "[info] trainer has no GPTQ_INSTA_CACHE hook; running stream-only"
  run_case "stream_seed${SEED}" "GPTQ_INSTA_CACHE=0"
fi

echo "[done] logs: ${RUN_DIR}"
for f in "${RUN_DIR}"/*.log; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  grep -nE "step:6500|stopping_early|gptq:calibrated|gptq:insta_cache|final_sliding_window_exact" "$f" | tail -n 20 || true
done
