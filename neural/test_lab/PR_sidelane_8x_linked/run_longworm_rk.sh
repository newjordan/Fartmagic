#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${SCRIPT_DIR}/sidelane"
RUNNER="${BUNDLE_ROOT}/scripts/run_longworm_rk_concepts_brotli.sh"
if [[ ! -x "${RUNNER}" ]]; then
  echo "Missing executable runner: ${RUNNER}" >&2
  echo "Re-pull TEST_LAB and confirm test_lab assets are present." >&2
  exit 1
fi

export PROJECT_CODENAME="${PROJECT_CODENAME:-longworm}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
exec bash "${RUNNER}" "$@"
