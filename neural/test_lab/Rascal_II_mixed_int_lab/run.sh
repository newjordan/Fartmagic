#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Walk up to find repo root (directory containing data/)
if [ -n "${REPO_ROOT:-}" ] && [ -d "${REPO_ROOT}/data/tokenizers" ]; then
  : # already set by caller
else
  DIR="${SCRIPT_DIR}"
  while [ "${DIR}" != "/" ]; do
    [ -d "${DIR}/data/tokenizers" ] && break
    DIR="$(dirname "${DIR}")"
  done
  if [ "${DIR}" = "/" ]; then
    echo "ERROR: could not find data/tokenizers/ in any parent directory" >&2
    exit 1
  fi
  REPO_ROOT="${DIR}"
fi
cd "${REPO_ROOT}"

# Find the training script relative to this script
TRAIN_SCRIPT="${SCRIPT_DIR}/../../experiments/Rascal_II_mixed_int_lab/train_gpt.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  # Fallback: nested under neural/ in parameter-golf-lab
  TRAIN_SCRIPT="${SCRIPT_DIR}/../../../neural/experiments/Rascal_II_mixed_int_lab/train_gpt.py"
fi

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  "${TRAIN_SCRIPT}" "$@"
