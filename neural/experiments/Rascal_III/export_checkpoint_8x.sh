#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

if [ -z "${INIT_MODEL_PATH:-}" ]; then
  echo "ERROR: set INIT_MODEL_PATH to the checkpoint you want to export/calibrate." >&2
  echo "Example: INIT_MODEL_PATH=/workspace/parameter-golf/final_model.pt SEED=444 NPROC_PER_NODE=8 bash neural/experiments/Rascal_III/export_checkpoint_8x.sh" >&2
  exit 1
fi
if [ ! -f "${INIT_MODEL_PATH}" ] && [ -f "${REPO_ROOT}/${INIT_MODEL_PATH}" ]; then
  INIT_MODEL_PATH="${REPO_ROOT}/${INIT_MODEL_PATH}"
fi
if [ ! -f "${INIT_MODEL_PATH}" ]; then
  echo "ERROR: INIT_MODEL_PATH not found: ${INIT_MODEL_PATH}" >&2
  exit 1
fi
export INIT_MODEL_PATH

export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export RUN_ID="${RUN_ID:-rascal_iii_ckpt_export_seed${SEED}}"

# Checkpoint-only mode: no training updates, no warmup, no averaging drift.
export ITERATIONS="${ITERATIONS:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-0}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export LAWA_ENABLED="${LAWA_ENABLED:-0}"

# Canonical loader behavior (same sampling config as the main runner).
export LOADER_MODE="${LOADER_MODE:-coprime}"
export COPRIME_SHARDS_PER_BATCH="${COPRIME_SHARDS_PER_BATCH:-1}"
export COPRIME_SHARD_HOLD_STEPS="${COPRIME_SHARD_HOLD_STEPS:-64}"

# Keep the same stable architecture knobs.
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MTP_NUM_HEADS="${MTP_NUM_HEADS:-0}"
export TRIGRAM="0"
export GATED_ATTENTION="0"
export VALUE_RESIDUAL="0"
export DTG_ENABLED="0"
export QAT_ENABLED="0"

# Export/calibration behavior.
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export SKIP_GPTQ="${SKIP_GPTQ:-0}"
export GPTQ_RESERVE_MS="${GPTQ_RESERVE_MS:-0}"
export QUANT_ATTN_BITS="${QUANT_ATTN_BITS:-5}"
export QUANT_MLP_BITS="${QUANT_MLP_BITS:-6}"
export QUANT_AUX_BITS="${QUANT_AUX_BITS:-6}"
export QUANT_EMBED_BITS="${QUANT_EMBED_BITS:-8}"
export QUANT_OTHER_BITS="${QUANT_OTHER_BITS:-8}"
export QUANT_ARTIFACT_PATH="${QUANT_ARTIFACT_PATH:-final_model.rascal_iii_ckpt_seed${SEED}.ptz}"

# Evaluation controls.
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
export NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-0}"

pip install brotli -q 2>/dev/null || true

TRAIN_SCRIPT="${REPO_ROOT}/neural/experiments/Rascal_III/train_gpt.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  TRAIN_SCRIPT="${REPO_ROOT}/experiments/Rascal_III/train_gpt.py"
fi

echo "rascal_iii_ckpt_export_script:${TRAIN_SCRIPT}"
echo "rascal_iii_ckpt_export_profile init=${INIT_MODEL_PATH} seed=${SEED} nproc=${NPROC_PER_NODE} iterations=${ITERATIONS} warmup_steps=${WARMUP_STEPS} skip_gptq=${SKIP_GPTQ} quant_bits=${QUANT_ATTN_BITS}/${QUANT_MLP_BITS}/${QUANT_AUX_BITS}/${QUANT_EMBED_BITS}/${QUANT_OTHER_BITS} skip_final_eval=${SKIP_FINAL_EVAL}"

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
