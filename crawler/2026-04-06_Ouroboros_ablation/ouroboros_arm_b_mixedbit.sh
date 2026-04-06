#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

export SEED="${SEED:-300}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MAX_WALLCLOCK_SECONDS=600
export WARMDOWN_ITERS=2000
export NUM_FLAT_LAYERS=9
export NUM_CRAWLER_LAYERS=1
export CRAWLER_LOOPS=2
export USE_CRAWLER=1
export COMPILE_FULLGRAPH=1
export SKIP_GPTQ=0
export LOOP_AWARE_GPTQ=1
export QK_GAIN_INIT=4.0
export GPTQ_CAL_SAMPLES=128
export GPTQ_CAL_SEQ_LEN=2048
export CRAWLER_LOOP_ROPE_SCALES=9,1,1
export SKIP_EMA=1
export MODEL_DIM=512
export INST_DIM=32
export CRAWLER_MLP_MULT=6.0
export CRAWLER_TAP_DIM=0
export ANCHOR_DIM=0
export CRAWLER_MLP_CHOKE_DIM=0
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2048
export ROPE_DIMS=16
export SWA_EVERY=50
export MATRIX_LR=0.03
export MLP_LEAKY_SLOPE=0.5
export CRAWLER_MLP_LEAKY_SLOPE=0.5
# ONE CHANGE: int8 for crawler blocks (reduce quant compounding through loops)
# Already wired in Ouroboros: crawler_blocks get int8, flat blocks stay int6
export CRAWLER_QUANT_INT8=1
export RUN_ID="ouroboros_arm_b_crawler_int8_seed${SEED}"

TRAIN_SCRIPT="$(find "${REPO_ROOT}" -path "*/Ouroboros_ablation/scripts/train_gpt.py" | head -1)"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
