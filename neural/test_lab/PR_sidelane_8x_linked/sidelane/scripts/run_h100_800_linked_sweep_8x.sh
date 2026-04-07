#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export RUNS_SUBDIR="${RUNS_SUBDIR:-h100_800_linked_8x}"

# Keep the same 800-step comparison profile unless explicitly overridden.
export ITERATIONS="${ITERATIONS:-800}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-2400}"

exec bash "${SCRIPT_DIR}/run_h100_800_linked_sweep.sh" "$@"
