#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Prefer local lab paths when available; allow caller override.
if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ -d "/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp8192" ]]; then
    export DATA_PATH="/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp8192"
  else
    export DATA_PATH="../../data/datasets/fineweb10B_sp8192"
  fi
fi
if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  if [[ -f "/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_8192_bpe.model" ]]; then
    export TOKENIZER_PATH="/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_8192_bpe.model"
  else
    export TOKENIZER_PATH="../../data/tokenizers/fineweb_8192_bpe.model"
  fi
fi

export SEED="${SEED:-4}"
export RUN_ID="${RUN_ID:-tone_comp_s${SEED}}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"

# TON-E architecture profile (our runner + his rhythm).
export USE_CRAWLER="${USE_CRAWLER:-1}"
export TON_E_RHYTHM="${TON_E_RHYTHM:-1}"
export TON_E_NUM_FLAT_LAYERS="${TON_E_NUM_FLAT_LAYERS:-4}"
export TON_E_NUM_CRAWLER_LAYERS="${TON_E_NUM_CRAWLER_LAYERS:-2}"
export TON_E_CRAWLER_LOOPS="${TON_E_CRAWLER_LOOPS:-2}"
export TON_E_XSA_INCLUDE_FLAT="${TON_E_XSA_INCLUDE_FLAT:-1}"
export EXPORT_QUANT="${EXPORT_QUANT:-int8_flat}"

# Competition-oriented quant/export defaults.
export SKIP_EMA="${SKIP_EMA:-1}"
export SKIP_GPTQ="${SKIP_GPTQ:-0}"
export LOOP_AWARE_GPTQ="${LOOP_AWARE_GPTQ:-1}"
export GPTQ_CAL_SAMPLES="${GPTQ_CAL_SAMPLES:-256}"
export CRAWLER_QUANT_INT8="${CRAWLER_QUANT_INT8:-0}"
export SIZE_TARGET_BYTES="${SIZE_TARGET_BYTES:-16000000}"
export SELECTIVE_PRUNE_ENABLE="${SELECTIVE_PRUNE_ENABLE:-1}"
export SELECTIVE_PRUNE_FACTOR="${SELECTIVE_PRUNE_FACTOR:-8}"
export COMPILE_ENABLED="${COMPILE_ENABLED:-1}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-1}"

mkdir -p logs artifacts

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
[TON-E dry-run]
cwd=${SCRIPT_DIR}
RUN_ID=${RUN_ID}
SEED=${SEED}
DATA_PATH=${DATA_PATH}
TOKENIZER_PATH=${TOKENIZER_PATH}
VOCAB_SIZE=${VOCAB_SIZE}
TON_E=${TON_E_NUM_FLAT_LAYERS}F+${TON_E_NUM_CRAWLER_LAYERS}Cx${TON_E_CRAWLER_LOOPS}
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}
GPTQ(loop-aware)=${LOOP_AWARE_GPTQ}
EXPORT_QUANT=${EXPORT_QUANT}
SIZE_TARGET_BYTES=${SIZE_TARGET_BYTES}
EOF
  exit 0
fi

WORLD_SIZE="${WORLD_SIZE:-1}"
if [[ "${WORLD_SIZE}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${WORLD_SIZE}" train_gpt.py
else
  python3 train_gpt.py
fi

LOG_PATH="logs/${RUN_ID}.txt"
ART_DIR="artifacts/seed_${SEED}"
mkdir -p "${ART_DIR}"
cp -f train_gpt.py "${ART_DIR}/train_gpt.py"
[[ -f final_model.pt ]] && cp -f final_model.pt "${ART_DIR}/"
[[ -f final_model.int6.ptz ]] && cp -f final_model.int6.ptz "${ART_DIR}/"
[[ -f final_model.int8.ptz ]] && cp -f final_model.int8.ptz "${ART_DIR}/"
[[ -f "${LOG_PATH}" ]] && cp -f "${LOG_PATH}" "${ART_DIR}/train_seed${SEED}.log"

echo "[TON-E] complete seed=${SEED} run_id=${RUN_ID}"
echo "[TON-E] artifacts => ${ART_DIR}"
