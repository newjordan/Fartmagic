#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

env \
  MODE=smoke \
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
  ITERATIONS="${ITERATIONS:-12000}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-240}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-393216}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}" \
  EVAL_STRIDE="${EVAL_STRIDE:-2048}" \
  SEED="${SEED:-444}" \
  bash "${SCRIPT_DIR}/run_ablation_sequence.sh"

