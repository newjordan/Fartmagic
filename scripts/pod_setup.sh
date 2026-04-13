#!/usr/bin/env bash
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Canonical entrypoint: keep implementation in Im_sorry_pod_setup.sh.
exec bash "${REPO_ROOT}/scripts/Im_sorry_pod_setup.sh"
