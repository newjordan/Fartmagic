#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Full-run wrapper for the legacy fast->full staged crawler profile.
# Defaults avoid 10-minute test cap and fullgraph recompile hard-fail.
export SEED="${SEED:-4}"
export RUN_ID="${RUN_ID:-tone_legacy_fast2full_full_s${SEED}}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export COMPILE_ENABLED="${COMPILE_ENABLED:-1}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"

bash "${SCRIPT_DIR}/run_test_legacy_fast2full.sh"
