#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# crawler_9f_corpus_ablations_v1 — 4xGPU 1500-step screen
#
# Usage:
#   SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-08_crawler_9f_corpus_ablations_v1/gate.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

env \
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
  ITERATIONS="${ITERATIONS:-1500}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3600}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-393216}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}" \
  EVAL_STRIDE="${EVAL_STRIDE:-64}" \
  SEED="${SEED:-444}" \
  bash "${SCRIPT_DIR}/run_ablation_sequence.sh"
