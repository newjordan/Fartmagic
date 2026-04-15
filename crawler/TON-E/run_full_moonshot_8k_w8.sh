#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Full-run wrapper: RWKV Moonshot 7F+3C, 8k vocab, no ramp, 8xH100.
export SEED="${SEED:-4}"
export WORLD_SIZE="${WORLD_SIZE:-8}"
export RUN_ID="${RUN_ID:-tone_moonshot_v8k_full_w${WORLD_SIZE}_s${SEED}}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

bash "${SCRIPT_DIR}/run_moonshot_8k.sh"
