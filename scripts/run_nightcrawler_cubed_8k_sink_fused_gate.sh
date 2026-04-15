#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 8k tokenizer/data defaults.
if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ -d "/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp8192" ]]; then
    export DATA_PATH="/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp8192"
  elif [[ -d "../../data/datasets/fineweb10B_sp8192" ]]; then
    export DATA_PATH="../../data/datasets/fineweb10B_sp8192"
  else
    echo "ERROR: 8k dataset path not found. Set DATA_PATH."
    exit 1
  fi
fi

if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  if [[ -f "/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_8192_bpe.model" ]]; then
    export TOKENIZER_PATH="/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_8192_bpe.model"
  elif [[ -f "../../data/tokenizers/fineweb_8192_bpe.model" ]]; then
    export TOKENIZER_PATH="../../data/tokenizers/fineweb_8192_bpe.model"
  else
    echo "ERROR: 8k tokenizer not found. Set TOKENIZER_PATH."
    exit 1
  fi
fi

# Nightcrawler Cubed 7F+3C — NO RAMP — clean gate test of sink + fused norm.
export SEED="${SEED:-4}"
export RUN_ID="${RUN_ID:-tone_nc3_v8k_sink_fused_gate_s${SEED}}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export MODEL_DIM="${MODEL_DIM:-512}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export TON_E_RHYTHM="${TON_E_RHYTHM:-0}"
export USE_CRAWLER="${USE_CRAWLER:-1}"
export NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-7}"
export NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-3}"
export CRAWLER_LOOPS="${CRAWLER_LOOPS:-3}"

# NEW FEATURES under test:
export SINK_TOKEN="${SINK_TOKEN:-1}"
export FUSED_NORM="${FUSED_NORM:-1}"

# NO ramp — crawler fully active from step 0 for clean signal measurement.
export CRAWLER_COMPUTE_STAGED="${CRAWLER_COMPUTE_STAGED:-0}"
export CRAWLER_GRADUATED="${CRAWLER_GRADUATED:-0}"
export CRAWLER_FORWARD_GRADUATED="${CRAWLER_FORWARD_GRADUATED:-0}"

# Standard export.
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

bash "${SCRIPT_DIR}/run_competition.sh"
