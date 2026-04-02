#!/usr/bin/env bash
set -euo pipefail

cd /home/frosty40/parameter-golf-lab

if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
else
  echo "FATAL: missing /venv/main/bin/activate" >&2
  exit 1
fi

export PYTHONPATH="/home/frosty40/parameter-golf-lab/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
mkdir -p /home/frosty40/parameter-golf-lab/logs
RUN_ID="rascal_lc4_s${SEED}_$(date +%Y%m%d_%H%M%S)"
LOG="/home/frosty40/parameter-golf-lab/logs/${RUN_ID}.log"

SEED="${SEED}" \
RUN_ID="${RUN_ID}" \
DATA_PATH="/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_1024_bpe.model" \
ITERATIONS=20000 \
WARMDOWN_ITERS=3500 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=4000 \
TRAIN_LOG_EVERY=500 \
COMPILE_ENABLED=1 \
COMPILE_FULLGRAPH=1 \
SKIP_GPTQ=1 \
LOADER_MODE=coprime \
COPRIME_MAX_LOADED_SHARDS=4 \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=0 \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
/home/frosty40/parameter-golf-lab/junkyard/experiments/Rascal_Final_Submission_LC4/train_gpt.py \
2>&1 | tee "${LOG}"
