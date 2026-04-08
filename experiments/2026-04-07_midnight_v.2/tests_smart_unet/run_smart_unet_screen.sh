#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="${ROOT}/experiments/2026-04-07_midnight_v.2/tests_smart_unet/run_mode_batch.sh"

SEED="${SEED:-444}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
PROFILE="${PROFILE:-screen}" \
PRIORITY_MAX="${PRIORITY_MAX:-1}" \
BUDGET_MINUTES="${BUDGET_MINUTES:-45}" \
REUSE_DONE="${REUSE_DONE:-1}" \
REUSE_MAX_AGE_MIN="${REUSE_MAX_AGE_MIN:-0}" \
STOP_ON_FAIL="${STOP_ON_FAIL:-1}" \
bash "${RUNNER}"
