#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# crawler_layer_relationship_ablation — 2xGPU 1000-step grid
#
# Usage:
#   SEED=444 NPROC_PER_NODE=2 bash crawler/2026-04-09_crawler_layer_relationship_ablation/gate.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

env \
  NPROC_PER_NODE="${NPROC_PER_NODE:-2}" \
  ITERATIONS="${ITERATIONS:-1000}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3600}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-196608}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}" \
  EVAL_STRIDE="${EVAL_STRIDE:-64}" \
  SEED="${SEED:-444}" \
  bash "${SCRIPT_DIR}/run_grid.sh"
