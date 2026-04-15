#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Auto-detect data/tokenizer paths. Check common pod locations.
if [[ -z "${DATA_PATH:-}" ]]; then
  for _d in "${SCRIPT_DIR}/../data/datasets/fineweb10B_sp8192" \
            "${SCRIPT_DIR}/../../data/datasets/fineweb10B_sp8192" \
            "/root/Fartmagic/data/datasets/fineweb10B_sp8192" \
            "/dev/shm/fineweb10B_sp8192" \
            "/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192" \
            "/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp8192"; do
    if [[ -d "$_d" ]]; then export DATA_PATH="$_d"; break; fi
  done
  [[ -z "${DATA_PATH:-}" ]] && { echo "FATAL: Cannot find fineweb10B_sp8192 dataset. Set DATA_PATH."; exit 1; }
fi
if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  for _t in "${SCRIPT_DIR}/../data/tokenizers/fineweb_8192_bpe.model" \
            "${SCRIPT_DIR}/../../data/tokenizers/fineweb_8192_bpe.model" \
            "/root/Fartmagic/data/tokenizers/fineweb_8192_bpe.model" \
            "/dev/shm/fineweb_8192_bpe.model" \
            "/workspace/SOTA_FINAL/data/tokenizers/fineweb_8192_bpe.model" \
            "/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_8192_bpe.model"; do
    if [[ -f "$_t" ]]; then export TOKENIZER_PATH="$_t"; break; fi
  done
  [[ -z "${TOKENIZER_PATH:-}" ]] && { echo "FATAL: Cannot find fineweb_8192_bpe.model tokenizer. Set TOKENIZER_PATH."; exit 1; }
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

# Auto-detect venv paths so torchrun/python3 always resolve.
for _p in /venv/main/bin /opt/conda/bin /usr/local/bin; do
  [[ -d "$_p" ]] && export PATH="$_p:$PATH"
done

WORLD_SIZE="${WORLD_SIZE:-1}"
if [[ "${WORLD_SIZE}" -gt 1 ]]; then
  python3 -m torch.distributed.run --standalone --nproc_per_node="${WORLD_SIZE}" train_gpt.py
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
