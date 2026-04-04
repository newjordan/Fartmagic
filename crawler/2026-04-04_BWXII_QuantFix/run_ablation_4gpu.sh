#!/bin/bash
set -euo pipefail
# ================================================================
# BW XII Quant Fix — 4-GPU Parallel Ablation
#
# Close the 0.049 quant gap on SplitHead architecture.
# Smart Skip + WD sweep + GPTQ tuning + stride + cross-head count
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

pip install brotli -q 2>/dev/null || true

# SplitHead base config (B6+WD=0.12 = our micro champion)
MICRO_ENV=(
    SEED="${SEED}"
    ITERATIONS=200
    MAX_WALLCLOCK_SECONDS=0
    WARMDOWN_ITERS=50
    TRAIN_BATCH_TOKENS=131072
    VAL_BATCH_SIZE=131072
    EVAL_STRIDE=2048
    TRAIN_SEQ_LEN=512
    EVAL_SEQ_LEN=512
    COMPILE_ENABLED=0
    COMPILE_FULLGRAPH=0
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=5
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=1
    CRAWLER_MLP_MULT=4.0
    MODEL_DIM=256
    NUM_HEADS=4
    NUM_KV_HEADS=2
    INST_DIM=16
    BIGRAM_VOCAB_SIZE=512
    BIGRAM_DIM=64
    XSA_LAST_N=0
    ROPE_DIMS=8
    SWA_EVERY=20
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
    HELIX=1
    HELIX_DIM=192
    HELIX_STRIDE=1
    CRAWLER_CROSS_HEADS=4
    MUON_WD=0.12
    TRAIN_LOG_EVERY=50
    VAL_LOSS_EVERY=200
)

run_arm() {
    local gpu="$1"; shift
    local tag="$1"; shift
    local logfile="${RESULTS_DIR}/${tag}_s${SEED}_${TS}.log"
    echo "[GPU${gpu}] ${tag} — $(date)"
    CUDA_VISIBLE_DEVICES="${gpu}" env "${MICRO_ENV[@]}" "$@" \
        python "${TRAIN_PY}" > "${logfile}" 2>&1
    local bpb=$(grep -oP 'step:200/200 val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local int6=$(grep -oP 'final_int6_roundtrip_exact.*val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local params=$(grep -oP 'model_params:\K[0-9]+' "${logfile}" 2>/dev/null || echo "?")
    local gap="?"
    if [[ "${bpb}" != "?" && "${int6}" != "?" ]]; then
        gap=$(python3 -c "print(f'{${int6}-${bpb}:.4f}')" 2>/dev/null || echo "?")
    fi
    echo "${tag}	${params}	${bpb}	${int6}	${gap}	${step_ms}" >> "${RESULTS_DIR}/gpu${gpu}_${TS}.tsv"
    echo "[GPU${gpu}] >>> ${tag}: bpb=${bpb} int6=${int6} gap=${gap} step_ms=${step_ms}"
}

run_gpu_batch() {
    local gpu="$1"; shift
    local batch_file="${RESULTS_DIR}/gpu${gpu}_${TS}.tsv"
    echo -e "arm\tparams\traw_bpb\tint6_bpb\tquant_gap\tstep_ms" > "${batch_file}"
    "$@"
    echo "[GPU${gpu}] BATCH COMPLETE"
}

echo ""
echo "================================================================"
echo "  BW XII QUANT FIX — 4-GPU Parallel Ablation"
echo "  Close the quant gap on SplitHead"
echo "  $(date)"
echo "================================================================"

# GPU 0: Smart Skip + controls (8 arms)
run_gpu_batch 0 bash -c "
$(declare -f run_arm)
$(declare -p MICRO_ENV SEED TRAIN_PY RESULTS_DIR TS)

# Controls
run_arm 0 Q0_splithead_ctrl SMART_SKIP=0
run_arm 0 Q1_no_helix HELIX=0 CRAWLER_CROSS_HEADS=0 MUON_WD=0.04 SMART_SKIP=0

# Smart Skip
run_arm 0 Q2_smart_skip SMART_SKIP=1
run_arm 0 Q3_smart_skip_wd015 SMART_SKIP=1 MUON_WD=0.15
run_arm 0 Q4_smart_skip_wd020 SMART_SKIP=1 MUON_WD=0.20

# Smart Skip + stride (faster steps = more steps + less quant pressure)
run_arm 0 Q5_smart_skip_s2 SMART_SKIP=1 HELIX_STRIDE=2
run_arm 0 Q6_smart_skip_s3 SMART_SKIP=1 HELIX_STRIDE=3

# Smart Skip + cross=2 (half cross — less shared-weight pressure)
run_arm 0 Q7_smart_skip_cross2 SMART_SKIP=1 CRAWLER_CROSS_HEADS=2
" &
PID0=$!

# GPU 1: Weight decay sweep (8 arms)
run_gpu_batch 1 bash -c "
$(declare -f run_arm)
$(declare -p MICRO_ENV SEED TRAIN_PY RESULTS_DIR TS)

# WD sweep without Smart Skip
run_arm 1 W0_wd008 MUON_WD=0.08 SMART_SKIP=0
run_arm 1 W1_wd010 MUON_WD=0.10 SMART_SKIP=0
run_arm 1 W2_wd012 MUON_WD=0.12 SMART_SKIP=0
run_arm 1 W3_wd015 MUON_WD=0.15 SMART_SKIP=0
run_arm 1 W4_wd020 MUON_WD=0.20 SMART_SKIP=0
run_arm 1 W5_wd025 MUON_WD=0.25 SMART_SKIP=0

# Adam WD on non-crawler params
run_arm 1 W6_adam_wd012 MUON_WD=0.12 ADAM_WD=0.12 SMART_SKIP=0
run_arm 1 W7_adam_wd020 MUON_WD=0.20 ADAM_WD=0.20 SMART_SKIP=0
" &
PID1=$!

# GPU 2: Stride + cross-head count (8 arms)
run_gpu_batch 2 bash -c "
$(declare -f run_arm)
$(declare -p MICRO_ENV SEED TRAIN_PY RESULTS_DIR TS)

# Stride sweep (fewer crawler fires = less shared-weight amplification)
run_arm 2 S0_stride1 HELIX_STRIDE=1 SMART_SKIP=0
run_arm 2 S1_stride2 HELIX_STRIDE=2 SMART_SKIP=0
run_arm 2 S2_stride3 HELIX_STRIDE=3 SMART_SKIP=0

# Cross-head count at production (less cross = smaller quant gap?)
run_arm 2 C0_cross4 CRAWLER_CROSS_HEADS=4 SMART_SKIP=0
run_arm 2 C1_cross2 CRAWLER_CROSS_HEADS=2 SMART_SKIP=0
run_arm 2 C2_cross3 CRAWLER_CROSS_HEADS=3 SMART_SKIP=0

# Bridge dim (smaller dim = less quant pressure?)
run_arm 2 D0_dim192 HELIX_DIM=192 SMART_SKIP=0
run_arm 2 D1_dim128 HELIX_DIM=128 SMART_SKIP=0
" &
PID2=$!

# GPU 3: Combos — stack best from each section (8 arms)
run_gpu_batch 3 bash -c "
$(declare -f run_arm)
$(declare -p MICRO_ENV SEED TRAIN_PY RESULTS_DIR TS)

# Smart Skip + high WD + stride=2 (triple quant fix)
run_arm 3 X0_triple_fix SMART_SKIP=1 MUON_WD=0.20 HELIX_STRIDE=2
run_arm 3 X1_triple_fix_cross2 SMART_SKIP=1 MUON_WD=0.20 HELIX_STRIDE=2 CRAWLER_CROSS_HEADS=2

# Smart Skip + dim=128 (smaller bridge = tighter weights)
run_arm 3 X2_smart_dim128 SMART_SKIP=1 HELIX_DIM=128
run_arm 3 X3_smart_dim128_s2 SMART_SKIP=1 HELIX_DIM=128 HELIX_STRIDE=2

# Crawler int8 (higher precision for shared block)
run_arm 3 X4_crawler_int8 CRAWLER_QUANT_INT8=1 SMART_SKIP=0
run_arm 3 X5_crawler_int8_smart CRAWLER_QUANT_INT8=1 SMART_SKIP=1

# The maximalist: everything
run_arm 3 X6_max SMART_SKIP=1 MUON_WD=0.20 HELIX_STRIDE=2 CRAWLER_QUANT_INT8=1
run_arm 3 X7_max_dim128 SMART_SKIP=1 MUON_WD=0.20 HELIX_STRIDE=2 CRAWLER_QUANT_INT8=1 HELIX_DIM=128
" &
PID3=$!

echo ""
echo "All 4 GPUs running. Waiting..."
echo "  GPU0 (PID ${PID0}): Smart Skip + controls"
echo "  GPU1 (PID ${PID1}): Weight decay sweep"
echo "  GPU2 (PID ${PID2}): Stride + cross-heads + bridge dim"
echo "  GPU3 (PID ${PID3}): Combos"
echo ""

wait $PID0; echo "GPU0 DONE — $(date)"
wait $PID1; echo "GPU1 DONE — $(date)"
wait $PID2; echo "GPU2 DONE — $(date)"
wait $PID3; echo "GPU3 DONE — $(date)"

# Merge
SUMMARY="${RESULTS_DIR}/quantfix_summary_${TS}.tsv"
echo -e "arm\tparams\traw_bpb\tint6_bpb\tquant_gap\tstep_ms" > "${SUMMARY}"
for g in 0 1 2 3; do
    tail -n +2 "${RESULTS_DIR}/gpu${g}_${TS}.tsv" >> "${SUMMARY}"
done

echo ""
echo "================================================================"
echo "  QUANT FIX ABLATION COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
