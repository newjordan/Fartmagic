#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# BW21_NoisyQAT_9F — 2-arm gate (control vs noisy_qat)
#
# Base: BWX 9F (NUM_FLAT_LAYERS=9, CRAWLER_LOOPS=3)
# Variable: NOISY_QAT=0 (control) vs NOISY_QAT=1 (treatment)
#
# Usage:
#   SEED=444 NPROC_PER_NODE=8 bash legs/2026-04-06_BW21_NoisyQAT_9F/gate.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
ITERATIONS=2000

mkdir -p "${SCRIPT_DIR}/results"
SUMMARY="${SCRIPT_DIR}/results/summary_s${SEED}_$(date +%Y%m%d_%H%M%S).tsv"

# Common env vars (BWX 9F defaults)
run_arm() {
    local arm_name="$1"
    local arm_desc="$2"
    local noisy_qat="$3"
    local log="${SCRIPT_DIR}/results/${arm_name}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "================================================================"
    echo "  ${arm_name}: ${arm_desc}"
    echo "  seed=${SEED}  GPUs=${NPROC}  steps=${ITERATIONS}"
    echo "  log: ${log}"
    echo "================================================================"
    echo ""

    if command -v torchrun >/dev/null 2>&1; then
        TORCHRUN=(torchrun)
    else
        TORCHRUN=(python3 -m torch.distributed.run)
    fi

    env \
        SEED="${SEED}" \
        MAX_WALLCLOCK_SECONDS=3600 \
        WARMDOWN_ITERS="${ITERATIONS}" \
        ITERATIONS="${ITERATIONS}" \
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
        NOISY_QAT="${noisy_qat}" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"

    # Extract metrics
    local raw_bpb int6_sw_bpb step_ms bytes_total model_params
    raw_bpb="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || true)"
    int6_sw_bpb="$(grep -oP 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}" | tail -1 || true)"
    bytes_total="$(grep -oP 'Total submission size int6\+(?:brotli|zstd|zlib): \K[0-9]+' "${log}" | tail -1 || true)"
    step_ms="$(grep -oP 'step_avg:\K[0-9.]+' "${log}" | tail -1 || true)"
    model_params="$(grep -oP 'model_params:\K[0-9]+' "${log}" | tail -1 || true)"

    echo ""
    echo "  >> ${arm_name}: params=${model_params:-?} raw=${raw_bpb:-?} int6_sw=${int6_sw_bpb:-?} step_ms=${step_ms:-?} bytes=${bytes_total:-?}"
    echo ""

    # Append to summary TSV
    if [[ ! -f "${SUMMARY}" ]]; then
        echo -e "arm\tdesc\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_ctrl" > "${SUMMARY}"
    fi
    echo -e "${arm_name}\t${arm_desc}\t${model_params:-?}\t${raw_bpb:-?}\t${int6_sw_bpb:-?}\t${step_ms:-?}\t${bytes_total:-?}\t?" >> "${SUMMARY}"
}

# ----------------------------------------------------------------
# ARM 0: Control (BWX 9F, no noisy QAT)
# ----------------------------------------------------------------
run_arm "BW21-00_ctrl" "BWX 9F control (NOISY_QAT=0)" "0"

# ----------------------------------------------------------------
# ARM 1: Treatment (BWX 9F + NOISY_QAT=1)
# ----------------------------------------------------------------
run_arm "BW21-01_nqat" "BWX 9F + NOISY_QAT=1" "1"

echo ""
echo "================================================================"
echo "  BW21_NoisyQAT_9F gate complete"
echo "  summary: ${SUMMARY}"
echo "================================================================"
cat "${SUMMARY}"
