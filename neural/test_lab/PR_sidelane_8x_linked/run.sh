#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NEURAL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${NEURAL_ROOT}/.." && pwd)"
PR_SIDELANE_GIT_URL="${PR_SIDELANE_GIT_URL:-https://github.com/newjordan/PR_sidelane.git}"
AUTO_CLONE_PR_SIDELANE="${AUTO_CLONE_PR_SIDELANE:-1}"

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
  if [[ "${AUTO_CLONE_PR_SIDELANE}" == "1" ]]; then
    bootstrap_root=""
    if [[ -d "/workspace" ]]; then
      bootstrap_root="/workspace/PR_sidelane"
    else
      bootstrap_root="${REPO_ROOT}/../PR_sidelane"
    fi

    if [[ -d "${bootstrap_root}/.git" ]]; then
      git -C "${bootstrap_root}" pull --ff-only origin main || true
    elif [[ ! -e "${bootstrap_root}" ]]; then
      echo "Bootstrapping linked sweep repo at ${bootstrap_root}" >&2
      git clone "${PR_SIDELANE_GIT_URL}" "${bootstrap_root}"
    fi

    path="${bootstrap_root}/scripts/run_h100_800_linked_sweep_8x.sh"
    if [[ -x "${path}" ]]; then
      PR_SIDELANE_ROOT="${bootstrap_root}"
      RUNNER="${path}"
    fi
  fi
fi

if [[ -z "${RUNNER}" ]]; then
  echo "Could not find executable runner: scripts/run_h100_800_linked_sweep_8x.sh" >&2
  echo "Tried roots:" >&2
  for root in "${candidate_roots[@]}"; do
    echo "  - ${root}" >&2
  done
  echo "Auto-bootstrap URL: ${PR_SIDELANE_GIT_URL}" >&2
  echo "Set PR_SIDELANE_ROOT=/path/to/PR_sidelane and rerun." >&2
  exit 1
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
exec bash "${RUNNER}" "$@"
