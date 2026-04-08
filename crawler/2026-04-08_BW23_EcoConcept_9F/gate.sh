#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

env \
  MODE=gate \
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
  ITERATIONS="${ITERATIONS:-2000}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3600}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}" \
  EVAL_STRIDE="${EVAL_STRIDE:-64}" \
  SEED="${SEED:-444}" \
  bash "${SCRIPT_DIR}/run_ablation_sequence.sh"

