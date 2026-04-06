#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-4096}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-4096}"
export EVAL_STRIDE="${EVAL_STRIDE:-128}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-393216}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"

export SSM_ENABLED="${SSM_ENABLED:-1}"
export SSM_IN_CRAWLER="${SSM_IN_CRAWLER:-1}"
export SSM_DIM="${SSM_DIM:-256}"
export SSM_KERNEL_SIZE="${SSM_KERNEL_SIZE:-4}"

echo "================================================================"
echo "  Helix_ab_3 — SSM + Long Context wrapper"
echo "  TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN} EVAL_SEQ_LEN=${EVAL_SEQ_LEN} EVAL_STRIDE=${EVAL_STRIDE}"
echo "  SSM_ENABLED=${SSM_ENABLED} SSM_DIM=${SSM_DIM} SSM_KERNEL_SIZE=${SSM_KERNEL_SIZE}"
echo "================================================================"

bash "${SCRIPT_DIR}/gate.sh"
