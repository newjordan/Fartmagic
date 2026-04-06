#!/usr/bin/env bash
set -euo pipefail
# ================================================================
#  Ouroboros Stacked — All 3 improvements combined
#
#  Noisy QAT + Crawler int8 + Contractive dt=0.5
#  Single 2k-step conflict check on 4×GPU.
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=4 bash neural/experiments/ouroboros_stacked/run_stacked.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt_stacked.py"

mkdir -p "${SCRIPT_DIR}/logs"
LOG="${SCRIPT_DIR}/logs/stacked_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "================================================================"
echo "  Ouroboros Stacked: NoisyQAT + CrawlerInt8 + Contractive"
echo "  seed=${SEED}  GPUs=${NPROC}  steps=2000"
echo "  log: ${LOG}"
echo "================================================================"
echo ""

env \
    SEED="${SEED}" \
    ITERATIONS=2000 \
    MAX_WALLCLOCK_SECONDS=3600 \
    WARMDOWN_ITERS=2000 \
    NUM_FLAT_LAYERS=9 \
    NUM_CRAWLER_LAYERS=1 \
    CRAWLER_LOOPS=2 \
    USE_CRAWLER=1 \
    COMPILE_FULLGRAPH=1 \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=1 \
    QK_GAIN_INIT=4.0 \
    GPTQ_CAL_SAMPLES=128 \
    GPTQ_CAL_SEQ_LEN=2048 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
    SKIP_EMA=1 \
    MODEL_DIM=512 \
    INST_DIM=32 \
    CRAWLER_MLP_MULT=6.0 \
    CRAWLER_TAP_DIM=0 \
    ANCHOR_DIM=0 \
    CRAWLER_MLP_CHOKE_DIM=0 \
    XSA_LAST_N=11 \
    BIGRAM_VOCAB_SIZE=2048 \
    ROPE_DIMS=16 \
    SWA_EVERY=50 \
    MATRIX_LR=0.03 \
    MLP_LEAKY_SLOPE=0.5 \
    CRAWLER_MLP_LEAKY_SLOPE=0.5 \
    NPROC_PER_NODE="${NPROC}" \
    NOISY_QAT=1 \
    CRAWLER_QUANT_INT8=1 \
    CRAWLER_CONTRACTIVE_DT=0.5 \
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
echo "  Ouroboros Stacked — Results (seed=${SEED})"
echo "  raw_bpb:     ${raw_bpb}"
echo "  int6_sw_bpb: ${int6_bpb}"
echo "  step_ms:     ${step_ms}"
echo "  bytes:       ${bytes}"
echo ""
echo "  Compare vs individual arms (seed=300 reference):"
echo "    control:     1.16409381"
echo "    noisy_qat:   1.16113228  (-0.00296)"
echo "    crawler_int8: 1.16094168 (-0.00315)"
echo "    contractive: 1.16182510  (-0.00227)"
echo "    sum of deltas:            -0.00838"
echo ""
echo "  If stacked < 1.161 → improvements compound"
echo "  If stacked > best individual (1.16094) → conflict detected"
echo "================================================================"
