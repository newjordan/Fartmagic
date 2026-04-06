#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NEURAL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${NEURAL_ROOT}/.." && pwd)"

candidate_roots=()
if [[ -n "${PR_SIDELANE_ROOT:-}" ]]; then
  candidate_roots+=("${PR_SIDELANE_ROOT}")
fi
candidate_roots+=(
  "${REPO_ROOT}/PR_sidelane"
  "${REPO_ROOT}/../PR_sidelane"
  "/workspace/PR_sidelane"
  "/home/frosty40/PR_sidelane"
)

RUNNER=""
for root in "${candidate_roots[@]}"; do
  path="${root}/scripts/run_h100_800_linked_sweep_8x.sh"
  if [[ -x "${path}" ]]; then
    PR_SIDELANE_ROOT="${root}"
    RUNNER="${path}"
    break
  fi
done

if [[ -z "${RUNNER}" ]]; then
  echo "Could not find executable runner: scripts/run_h100_800_linked_sweep_8x.sh" >&2
  echo "Tried roots:" >&2
  for root in "${candidate_roots[@]}"; do
    echo "  - ${root}" >&2
  done
  echo "Set PR_SIDELANE_ROOT=/path/to/PR_sidelane and rerun." >&2
  exit 1
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
exec bash "${RUNNER}" "$@"
