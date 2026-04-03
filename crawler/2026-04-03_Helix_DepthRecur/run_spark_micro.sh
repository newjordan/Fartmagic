#!/bin/bash
set -euo pipefail
# ================================================================
# Helix + Depth Recurrence — DGX Spark Micro Tests
#
# Tests how flat-layer depth recurrence interacts with the crawler.
# Does the crawler benefit from richer flat representations?
#
# Usage:
#   cd ~/parameter-golf-lab && bash crawler/2026-04-03_Helix_DepthRecur/run_spark_micro.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
TS="$(date +%Y%m%d_%H%M%S)"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/micro_summary_${TS}.tsv"

pip install brotli -q 2>/dev/null || true

# Production-scale config on 4×H100: real model size, compile on, 500 steps
MICRO_ENV=(
    SEED="${SEED}"
    ITERATIONS=500
    MAX_WALLCLOCK_SECONDS=0
    WARMDOWN_ITERS=100
    TRAIN_BATCH_TOKENS=786432
    VAL_BATCH_SIZE=524288
    EVAL_STRIDE=64
    TRAIN_SEQ_LEN=2048
    EVAL_SEQ_LEN=2048
    COMPILE_ENABLED=1
    COMPILE_FULLGRAPH=1
    TORCHDYNAMO_OPTIMIZE_DDP=0
    USE_CRAWLER=1
    NUM_CRAWLER_LAYERS=1
    CRAWLER_MLP_MULT=6.0
    MODEL_DIM=512
    NUM_HEADS=8
    NUM_KV_HEADS=4
    INST_DIM=32
    BIGRAM_VOCAB_SIZE=2048
    BIGRAM_DIM=128
    XSA_LAST_N=11
    ROPE_DIMS=16
    SWA_EVERY=50
    SKIP_GPTQ=1
    SKIP_EMA=1
    QK_GAIN_INIT=4.0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_TAP_DIM=0
    ANCHOR_DIM=0
    MATRIX_LR=0.03
    EMBED_LR=0.035
    HELIX_DIM=64
    TRAIN_LOG_EVERY=100
    VAL_LOSS_EVERY=500
    NPROC_PER_NODE="${NPROC}"
)

run_arm() {
    local tag="$1"; shift
    local logfile="${RESULTS_DIR}/${tag}_s${SEED}_${TS}.log"
    echo ""
    echo "================================================================"
    echo "  ARM: ${tag} — $(date)"
    echo "================================================================"
    env "${MICRO_ENV[@]}" "$@" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" 2>&1 | tee "${logfile}"
    local bpb=$(grep -oP 'step:500/500 val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local int6=$(grep -oP 'final_int6_sliding_window_exact.*val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local params=$(grep -oP 'model_params:\K[0-9]+' "${logfile}" 2>/dev/null || echo "?")
    echo -e "${tag}\t${params}\t${bpb}\t${int6}\t${step_ms}" >> "${SUMMARY}"
    echo "  >>> ${tag}: bpb=${bpb} int6=${int6} step_ms=${step_ms} params=${params}"
}

echo -e "arm\tparams\traw_bpb\tint6_sw_bpb\tstep_ms" > "${SUMMARY}"

echo ""
echo "================================================================"
echo "  HELIX + DEPTH RECURRENCE — 4×H100 PRODUCTION SCALE"
echo "  500 steps, dim=512, seq=2048, compile=on, ${NPROC} GPUs"
echo "================================================================"

# ----------------------------------------------------------------
# CONTROLS
# ----------------------------------------------------------------

echo ""
echo "==== CONTROLS ===="

# R0: 5F no helix, no recurrence (baseline)
run_arm "R0_ctrl_5f" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=0 RECUR_LAYERS=""

# R1: 5F no helix, with recurrence on layers 2,3 (field standard approach)
run_arm "R1_recur_only_5f_L23" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=0 RECUR_LAYERS=2,3

# R2: 5F helix dim=64, no recurrence (our best from suite 1)
run_arm "R2_helix_only_5f" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 RECUR_LAYERS=""

# ----------------------------------------------------------------
# HELIX + DEPTH RECURRENCE COMBOS
# ----------------------------------------------------------------

echo ""
echo "==== HELIX + DEPTH RECURRENCE — Does it stack? ===="

# S0: 5F helix + recur layers 2,3 (middle encoder layers)
run_arm "S0_helix_recur_L23" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 RECUR_LAYERS=2,3

# S1: 5F helix + recur layer 1 (early encoder)
run_arm "S1_helix_recur_L1" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 RECUR_LAYERS=1

# S2: 5F helix + recur layer 3 (late encoder / early decoder boundary)
run_arm "S2_helix_recur_L3" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 RECUR_LAYERS=3

# S3: 5F helix + recur layers 1,2,3 (heavy recurrence)
run_arm "S3_helix_recur_L123" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 RECUR_LAYERS=1,2,3

# ----------------------------------------------------------------
# DEPTH RECURRENCE AT 7F AND 9F
# ----------------------------------------------------------------

echo ""
echo "==== DEPTH SCALING + RECURRENCE ===="

# T0: 7F helix + recur layers 3,4 (middle of encoder)
run_arm "T0_7f_helix_recur_L34" \
    NUM_FLAT_LAYERS=7 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 RECUR_LAYERS=3,4

# T1: 7F no helix, recur layers 3,4 (isolate recurrence from helix)
run_arm "T1_7f_recur_only_L34" \
    NUM_FLAT_LAYERS=7 CRAWLER_LOOPS=1 HELIX=0 RECUR_LAYERS=3,4

# T2: 9F helix + recur layers 4,5 (matches field standard layer choice)
run_arm "T2_9f_helix_recur_L45" \
    NUM_FLAT_LAYERS=9 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 RECUR_LAYERS=4,5

# T3: 9F no helix, recur layers 4,5 (isolate)
run_arm "T3_9f_recur_only_L45" \
    NUM_FLAT_LAYERS=9 CRAWLER_LOOPS=1 HELIX=0 RECUR_LAYERS=4,5

# ----------------------------------------------------------------
# DELAYED START (field trick from PR #1296)
# ----------------------------------------------------------------

echo ""
echo "==== DELAYED RECURRENCE START ===="

# U0: 5F helix + recur L2,3 starting at step 125 (25% through training)
run_arm "U0_helix_recur_delayed125" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 RECUR_LAYERS=2,3 RECUR_START_STEP=125

# U1: 5F helix + recur L2,3 starting at step 250 (50% through)
run_arm "U1_helix_recur_delayed250" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 RECUR_LAYERS=2,3 RECUR_START_STEP=250

# ----------------------------------------------------------------
# DONE
# ----------------------------------------------------------------

echo ""
echo "================================================================"
echo "  DEPTH RECURRENCE MICRO TESTS COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
