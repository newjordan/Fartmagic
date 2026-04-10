#!/usr/bin/env bash
# Gate: 1-GPU, 2000 steps — BW_9F2C (NUM_CRAWLER_LAYERS=2)
# Only diff from BWX 9F parent: NUM_CRAWLER_LAYERS=2
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-444}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-200}"

env \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
    WARMDOWN_ITERS=2000 \
    COMPLEMENT_ALPHA=0 \
    XSA_LAST_N=11 \
    BIGRAM_VOCAB_SIZE=2048 \
    ROPE_DIMS=16 \
    SWA_EVERY=50 \
    MTP_NUM_HEADS=0 \
    LATE_QAT_THRESHOLD=0 \
    MATRIX_LR=0.03 \
    TORCHDYNAMO_OPTIMIZE_DDP=0 \
    COMPILE_FULLGRAPH=1 \
    NGRAM_EVAL_ORDER=0 \
    MODEL_DIM=512 \
    USE_CRAWLER=1 \
    NUM_FLAT_LAYERS=9 \
    NUM_CRAWLER_LAYERS=2 \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=6.0 \
    INST_DIM=32 \
    CRAWLER_QUANT_INT8=0 \
    DELTA_NET_HEADS=0 \
    SKIP_EMA=1 \
    SKIP_GPTQ=1 \
    LOOP_AWARE_GPTQ=0 \
    MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_CHOKE_DIM=0 \
    CRAWLER_MLP_CHOKE_SHAPE=flat \
    CRAWLER_MLP_CHOKE_GROUPS=8 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
    CRAWLER_LOOP_SMEAR=0 \
    CRAWLER_TAP_DIM=0 \
    CRAWLER_TAP_LOOP_SPECIFIC=1 \
    CRAWLER_TAP_LAYERS=all \
    ANCHOR_DIM=0 \
    FLAT_WEIGHT_SHARE=0 \
    SKIP_FINAL_EVAL=1 \
    python3 -m torch.distributed.run --standalone --nproc_per_node=1 \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "${SCRIPT_DIR}/gate_seed${SEED}.log"

echo "--- gate done. check step_avg and loss trend before proceeding to run.sh ---"
