#!/bin/bash
set -euo pipefail
# ================================================================
#  Ouroboros III — BWX 9F base + stacked improvements
#
#  Base: EXACT BWX 9F config (CRAWLER_LOOPS=3, SKIP_GPTQ=1)
#  Added: NoisyQAT, CrawlerInt8, Contractive dt=0.5, Mixed-bit (attn=5)
#         Loop-aware GPTQ (required for mixed-bit), Brotli compression
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-06_Ouroboros_III/run.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt_ouroboros_iii.py"

mkdir -p "${SCRIPT_DIR}/logs"
LOG="${SCRIPT_DIR}/logs/ouro_iii_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "================================================================"
echo "  Ouroboros III — BWX 9F base + all improvements"
echo "  NoisyQAT + CrawlerInt8 + Contractive + Mixed-bit + GPTQ + Brotli"
echo "  seed=${SEED}  GPUs=${NPROC}  wallclock=600s  LOOPS=3"
echo "  log: ${LOG}"
echo "================================================================"
echo ""

# ---------------------------------------------------------------
# ENV VARS: BWX 9F exact base (lines match records/.../run.sh)
# + intentional additions marked with # ADDED
# ---------------------------------------------------------------
env \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS=600 \
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
    NUM_CRAWLER_LAYERS=1 \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=6.0 \
    INST_DIM=32 \
    DELTA_NET_HEADS=0 \
    SKIP_EMA=1 \
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
    NPROC_PER_NODE="${NPROC}" \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=1 \
    GPTQ_CAL_SAMPLES=128 \
    GPTQ_CAL_SEQ_LEN=2048 \
    NOISY_QAT=1 \
    CRAWLER_QUANT_INT8=1 \
    CRAWLER_CONTRACTIVE_DT=0.5 \
    QUANT_ATTN_BITS=5 \
    QUANT_MLP_BITS=6 \
    QUANT_AUX_BITS=6 \
    QUANT_EMBED_BITS=8 \
    QUANT_OTHER_BITS=8 \
    torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
    2>&1 | tee "${LOG}"

# ----------------------------------------------------------------
# Extract results
# ----------------------------------------------------------------
int6_bpb=$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
raw_bpb=$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "${LOG}" | tail -1 || echo "?")
bytes=$(grep -oP 'Total submission size int6\+(?:zstd|zlib|brotli): \K[0-9]+' "${LOG}" | tail -1 || echo "?")

echo ""
echo "================================================================"
echo "  Ouroboros III — Results (seed=${SEED})"
echo "  raw_bpb:     ${raw_bpb}"
echo "  int6_sw_bpb: ${int6_bpb}"
echo "  step_ms:     ${step_ms}"
echo "  bytes:       ${bytes}"
echo ""
echo "  BWX 9F baseline to beat: 1.13867894 BPB, 15.2MB"
echo "  Must be under 16MB (16777216 bytes)"
echo "================================================================"
