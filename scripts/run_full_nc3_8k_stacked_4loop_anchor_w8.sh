#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Full-run wrapper: NC3 7F+3C stacked (4-loop ROPE + anchor), no ramp, 8xH100.
export SEED="${SEED:-4}"
export WORLD_SIZE="${WORLD_SIZE:-8}"
export RUN_ID="${RUN_ID:-tone_nc3_v8k_stacked_4loop_anchor_full_w8_s${SEED}}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

bash "${SCRIPT_DIR}/run_nc3_8k_stacked_4loop_anchor.sh"
