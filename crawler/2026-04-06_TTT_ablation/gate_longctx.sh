#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-4096}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-4096}"
export EVAL_STRIDE="${EVAL_STRIDE:-128}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-393216}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"

echo "================================================================"
echo "  TTT_ablation — Long Context wrapper"
echo "  TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN} EVAL_SEQ_LEN=${EVAL_SEQ_LEN} EVAL_STRIDE=${EVAL_STRIDE}"
echo "================================================================"

bash "${SCRIPT_DIR}/gate.sh"
