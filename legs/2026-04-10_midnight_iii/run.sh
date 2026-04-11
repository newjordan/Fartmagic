#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${LEG_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${LEG_DIR}/train_gpt.py"
LOG_DIR="${LEG_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/screen_seed${SEED:-444}_$(date +%Y%m%d_%H%M%S).log"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"

SEED="${SEED}" \
NPROC_PER_NODE="${NPROC}" \
torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
2>&1 | tee "${LOG_FILE}"

echo "LOG: ${LOG_FILE}"
