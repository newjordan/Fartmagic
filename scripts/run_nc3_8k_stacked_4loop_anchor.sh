#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Nightcrawler Cubed 7F+3C stacked: 4-loop differentiated ROPE + anchor.
# No ramp — crawler active from step 0.
# Stacking vectors validated individually on 9F+1C screen scale:
#   4-loop ROPE(9,3,1,1): -0.0046
#   ANCHOR_DIM=32:        -0.0024

# 8k tokenizer/data defaults. Discovery continues in run_competition.sh.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/resolve_sp8192_paths.sh"
fm_resolve_sp8192 "${SCRIPT_DIR}"

export SEED="${SEED:-4}"
export RUN_ID="${RUN_ID:-tone_nc3_v8k_stacked_4loop_anchor_s${SEED}}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export MODEL_DIM="${MODEL_DIM:-512}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export TON_E_RHYTHM="${TON_E_RHYTHM:-0}"
export USE_CRAWLER="${USE_CRAWLER:-1}"
export NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-7}"
export NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-3}"

# Stack vector 1: 4-loop differentiated ROPE battery
export CRAWLER_LOOPS="${CRAWLER_LOOPS:-4}"
export CRAWLER_LOOP_ROPE_SCALES="${CRAWLER_LOOP_ROPE_SCALES:-9,3,1,1}"

# Stack vector 2: Anchor projection
export ANCHOR_DIM="${ANCHOR_DIM:-32}"

# No staged ramp — crawler active from step 0.

# Export
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
