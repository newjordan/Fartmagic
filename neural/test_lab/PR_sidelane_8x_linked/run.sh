#!/usr/bin/env bash
set -euo pipefail

PR_SIDELANE_ROOT="${PR_SIDELANE_ROOT:-/home/frosty40/PR_sidelane}"
RUNNER="${PR_SIDELANE_ROOT}/scripts/run_h100_800_linked_sweep_8x.sh"

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner not found or not executable: ${RUNNER}" >&2
  exit 1
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
exec bash "${RUNNER}" "$@"
