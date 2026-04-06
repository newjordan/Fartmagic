#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  experiments/Rascal_II_homebase/train_gpt.py "$@"
