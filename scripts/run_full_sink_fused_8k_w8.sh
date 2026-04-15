#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export SEED="${SEED:-4}"
export WORLD_SIZE="${WORLD_SIZE:-8}"
export RUN_ID="${RUN_ID:-tone_nc3_v8k_sink_fused_full_w${WORLD_SIZE}_s${SEED}}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

bash "${SCRIPT_DIR}/run_nightcrawler_cubed_8k_sink_fused_gate.sh"
