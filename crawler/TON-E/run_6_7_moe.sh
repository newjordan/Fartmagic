#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 8k tokenizer/data defaults. Caller can override DATA_PATH/TOKENIZER_PATH/VOCAB_SIZE explicitly.
if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ -d "/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp8192" ]]; then
    export DATA_PATH="/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp8192"
  elif [[ -d "../../data/datasets/fineweb10B_sp8192" ]]; then
    export DATA_PATH="../../data/datasets/fineweb10B_sp8192"
  else
    echo "ERROR: 8k dataset path not found. Set DATA_PATH to your fineweb10B_sp8192 directory."
    exit 1
  fi
fi

if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  if [[ -f "/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_8192_bpe.model" ]]; then
    export TOKENIZER_PATH="/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_8192_bpe.model"
  elif [[ -f "../../data/tokenizers/fineweb_8192_bpe.model" ]]; then
    export TOKENIZER_PATH="../../data/tokenizers/fineweb_8192_bpe.model"
  else
    echo "ERROR: 8k tokenizer not found. Set TOKENIZER_PATH to fineweb_8192_bpe.model."
    exit 1
  fi
fi

# 6F+3C MoE variant — identical config to run_6_7.sh but with MoE gating on crawler loops.
export SEED="${SEED:-4}"
export RUN_ID="${RUN_ID:-tone_6_7_moe_v8k_sink_fused_s${SEED}}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export MODEL_DIM="${MODEL_DIM:-512}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export TON_E_RHYTHM="${TON_E_RHYTHM:-0}"
export USE_CRAWLER="${USE_CRAWLER:-1}"
export NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-6}"
export NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-3}"
export CRAWLER_LOOPS="${CRAWLER_LOOPS:-3}"
export CRAWLER_LOOP_ROPE_SCALES="${CRAWLER_LOOP_ROPE_SCALES:-9,1,1}"

# MoE gating — per-token soft weighting of crawler block contributions per loop.
export CRAWLER_MOE="${CRAWLER_MOE:-1}"
export CRAWLER_MOE_HIDDEN="${CRAWLER_MOE_HIDDEN:-16}"

# Sink token — learnable attention dump target for crawler loop stability.
export SINK_TOKEN="${SINK_TOKEN:-1}"
# Fused RMSNorm — residual add + norm in one torch.compile-fused pass.
export FUSED_NORM="${FUSED_NORM:-1}"

# Speed-first stepped crawler:
# - crawler stays off until 33% of run
# - crawler reaches full strength by 66% of run
export CRAWLER_COMPUTE_STAGED="${CRAWLER_COMPUTE_STAGED:-1}"
export CRAWLER_SAFE_WARMUP_STEPS="${CRAWLER_SAFE_WARMUP_STEPS:-0}"
export CRAWLER_GRADUATED="${CRAWLER_GRADUATED:-1}"
export CRAWLER_START_FRAC="${CRAWLER_START_FRAC:-0.0}"
export CRAWLER_RAMP_DELAY_FRAC="${CRAWLER_RAMP_DELAY_FRAC:-0.33}"
export CRAWLER_RAMP_FRAC="${CRAWLER_RAMP_FRAC:-0.33}"
export CRAWLER_FORWARD_GRADUATED="${CRAWLER_FORWARD_GRADUATED:-1}"
export CRAWLER_FORWARD_START_FRAC="${CRAWLER_FORWARD_START_FRAC:-0.0}"
export CRAWLER_FORWARD_RAMP_DELAY_FRAC="${CRAWLER_FORWARD_RAMP_DELAY_FRAC:-0.33}"
export CRAWLER_FORWARD_RAMP_FRAC="${CRAWLER_FORWARD_RAMP_FRAC:-0.33}"

# Keep Nightcrawler-style export path.
export EXPORT_QUANT="${EXPORT_QUANT:-int6}"
export SKIP_EMA="${SKIP_EMA:-1}"
export SKIP_GPTQ="${SKIP_GPTQ:-0}"
export LOOP_AWARE_GPTQ="${LOOP_AWARE_GPTQ:-1}"
export GPTQ_CAL_SAMPLES="${GPTQ_CAL_SAMPLES:-256}"
export SIZE_TARGET_BYTES="${SIZE_TARGET_BYTES:-16000000}"
export SELECTIVE_PRUNE_ENABLE="${SELECTIVE_PRUNE_ENABLE:-1}"
export SELECTIVE_PRUNE_FACTOR="${SELECTIVE_PRUNE_FACTOR:-8}"
export COMPILE_ENABLED="${COMPILE_ENABLED:-1}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"

mkdir -p logs artifacts

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
[TON-E MoE dry-run]
cwd=${SCRIPT_DIR}
RUN_ID=${RUN_ID}
SEED=${SEED}
DATA_PATH=${DATA_PATH}
TOKENIZER_PATH=${TOKENIZER_PATH}
VOCAB_SIZE=${VOCAB_SIZE}
ARCH=${NUM_FLAT_LAYERS}F+${NUM_CRAWLER_LAYERS}Cx${CRAWLER_LOOPS}
CRAWLER_MOE=${CRAWLER_MOE}
CRAWLER_MOE_HIDDEN=${CRAWLER_MOE_HIDDEN}
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}
GPTQ(loop-aware)=${LOOP_AWARE_GPTQ}
EXPORT_QUANT=${EXPORT_QUANT}
SIZE_TARGET_BYTES=${SIZE_TARGET_BYTES}
EOF
  exit 0
fi

# Launch training with the MoE training script.
WORLD_SIZE="${WORLD_SIZE:-1}"
if [[ "${WORLD_SIZE}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${WORLD_SIZE}" train_gpt_moe.py
else
  python3 train_gpt_moe.py
fi

LOG_PATH="logs/${RUN_ID}.txt"
ART_DIR="artifacts/seed_${SEED}"
mkdir -p "${ART_DIR}"
cp -f train_gpt_moe.py "${ART_DIR}/train_gpt_moe.py"
[[ -f final_model.pt ]] && cp -f final_model.pt "${ART_DIR}/"
[[ -f final_model.int6.ptz ]] && cp -f final_model.int6.ptz "${ART_DIR}/"
[[ -f final_model.int8.ptz ]] && cp -f final_model.int8.ptz "${ART_DIR}/"
[[ -f "${LOG_PATH}" ]] && cp -f "${LOG_PATH}" "${ART_DIR}/train_seed${SEED}.log"

echo "[TON-E MoE] complete seed=${SEED} run_id=${RUN_ID}"
echo "[TON-E MoE] artifacts => ${ART_DIR}"
