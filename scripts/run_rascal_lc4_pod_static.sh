#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-300}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="rascal_lc4_s${SEED}_${TS}"
LOG_DIR="${REPO_ROOT}/logs"
TRAINER="${REPO_ROOT}/junkyard/experiments/Rascal_Final_Submission_LC4/train_gpt.py"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="/usr/local/lib/python3.12/dist-packages:${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
export DATA_PATH="${REPO_ROOT}/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model"
export ITERATIONS=20000
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=4000
export TRAIN_LOG_EVERY=500
export COMPILE_ENABLED=1
export COMPILE_FULLGRAPH=1
export SKIP_GPTQ=1
export LOADER_MODE=coprime
export COPRIME_MAX_LOADED_SHARDS=4
export COPRIME_SHARDS_PER_BATCH=1
export COPRIME_SHARD_HOLD_STEPS=64
export COMPLEMENT_ALPHA=0
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2048
export ROPE_DIMS=16
export SWA_EVERY=50
export MTP_NUM_HEADS=0
export TRIGRAM=0
export NGRAM_EVAL_ORDER=0
export CUBRIC_CADENCE=0
export NGRAM_ENTROPY_SHIFT=0
export SEED
export RUN_ID

python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${TRAINER}" 2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
