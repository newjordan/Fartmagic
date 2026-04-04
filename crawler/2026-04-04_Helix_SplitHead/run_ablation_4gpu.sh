#!/bin/bash
set -euo pipefail
# ================================================================
# Helix SplitHead — 4-GPU Parallel Ablation
# Splits 30 arms across 4 GPUs running simultaneously
# ~40 min instead of ~2.5 hours
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
    HELIX_DIM=64
    HELIX_STRIDE=1
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
    local step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local params=$(grep -oP 'model_params:\K[0-9]+' "${logfile}" 2>/dev/null || echo "?")
    echo "${tag}	${params}	${bpb}	${step_ms}" >> "${RESULTS_DIR}/gpu${gpu}_${TS}.tsv"
    echo "[GPU${gpu}] >>> ${tag}: bpb=${bpb} step_ms=${step_ms}"
}

run_gpu_batch() {
    local gpu="$1"; shift
    local batch_file="${RESULTS_DIR}/gpu${gpu}_${TS}.tsv"
    echo -e "arm\tparams\traw_bpb\tstep_ms" > "${batch_file}"
    # Each function call below runs sequentially on this GPU
    "$@"
    echo "[GPU${gpu}] BATCH COMPLETE"
}

echo ""
echo "================================================================"
echo "  HELIX SPLITHEAD — 4-GPU Parallel Ablation"
echo "  30 arms across 4 GPUs, ~40 min total"
echo "  $(date)"
echo "================================================================"

# GPU 0: Controls + Split-head sweep (8 arms)
run_gpu_batch 0 bash -c "
$(declare -f run_arm)
$(declare -p MICRO_ENV SEED TRAIN_PY RESULTS_DIR TS)
run_arm 0 S0_helix_ctrl CRAWLER_CROSS_HEADS=0
run_arm 0 S1_no_helix HELIX=0 CRAWLER_CROSS_HEADS=0
run_arm 0 H1_cross1 CRAWLER_CROSS_HEADS=1
run_arm 0 H2_cross2 CRAWLER_CROSS_HEADS=2
run_arm 0 H3_cross3 CRAWLER_CROSS_HEADS=3
run_arm 0 H4_cross4_full CRAWLER_CROSS_HEADS=4
run_arm 0 D1_cross2_dim32 CRAWLER_CROSS_HEADS=2 HELIX_DIM=32
run_arm 0 D2_cross2_dim128 CRAWLER_CROSS_HEADS=2 HELIX_DIM=128
" &
PID0=$!

# GPU 1: Bridge dim + Competition tech (8 arms)
run_gpu_batch 1 bash -c "
$(declare -f run_arm)
$(declare -p MICRO_ENV SEED TRAIN_PY RESULTS_DIR TS)
run_arm 1 D3_cross4_dim32 CRAWLER_CROSS_HEADS=4 HELIX_DIM=32
run_arm 1 D4_cross4_dim128 CRAWLER_CROSS_HEADS=4 HELIX_DIM=128
run_arm 1 W1_wd009 CRAWLER_CROSS_HEADS=2 MUON_WD=0.09
run_arm 1 W2_wd012 CRAWLER_CROSS_HEADS=2 MUON_WD=0.12
run_arm 1 Q1_qk5 CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=5.0
run_arm 1 Q2_qk6 CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=6.0
run_arm 1 L1_lr002 CRAWLER_CROSS_HEADS=2 MATRIX_LR=0.02
run_arm 1 L2_lr004 CRAWLER_CROSS_HEADS=2 MATRIX_LR=0.04
" &
PID1=$!

# GPU 2: Crawler MLP + Depth + RoPE (8 arms)
run_gpu_batch 2 bash -c "
$(declare -f run_arm)
$(declare -p MICRO_ENV SEED TRAIN_PY RESULTS_DIR TS)
run_arm 2 M1_crawl_mlp6 CRAWLER_CROSS_HEADS=2 CRAWLER_MLP_MULT=6.0
run_arm 2 M2_crawl_mlp2 CRAWLER_CROSS_HEADS=2 CRAWLER_MLP_MULT=2.0
run_arm 2 M3_7f_cross2 CRAWLER_CROSS_HEADS=2 NUM_FLAT_LAYERS=7
run_arm 2 K1_7f_cross2_dim64 CRAWLER_CROSS_HEADS=2 NUM_FLAT_LAYERS=7
run_arm 2 K2_7f_cross4 CRAWLER_CROSS_HEADS=4 NUM_FLAT_LAYERS=7
run_arm 2 K3_7f_helix_ctrl CRAWLER_CROSS_HEADS=0 NUM_FLAT_LAYERS=7
run_arm 2 R1_rope_1_1_1 CRAWLER_CROSS_HEADS=2 CRAWLER_LOOP_ROPE_SCALES=1,1,1
run_arm 2 R2_rope_16_4_1 CRAWLER_CROSS_HEADS=2 CRAWLER_LOOP_ROPE_SCALES=16,4,1
" &
PID2=$!

# GPU 3: Combo stacking (6 arms — finishes faster, that's fine)
run_gpu_batch 3 bash -c "
$(declare -f run_arm)
$(declare -p MICRO_ENV SEED TRAIN_PY RESULTS_DIR TS)
run_arm 3 B1_cross2_qk5_wd09 CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=5.0 MUON_WD=0.09
run_arm 3 B2_cross2_qk5_dim128_7f CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=5.0 HELIX_DIM=128 NUM_FLAT_LAYERS=7
run_arm 3 B3_full_cross_qk5_wd09_dim128 CRAWLER_CROSS_HEADS=4 QK_GAIN_INIT=5.0 MUON_WD=0.09 HELIX_DIM=128
run_arm 3 B4_kitchen_sink CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=5.0 MUON_WD=0.09 HELIX_DIM=128 NUM_FLAT_LAYERS=7 CRAWLER_MLP_MULT=6.0
run_arm 3 B5_cross2_dim192 CRAWLER_CROSS_HEADS=2 HELIX_DIM=192
run_arm 3 B6_cross4_dim192 CRAWLER_CROSS_HEADS=4 HELIX_DIM=192
" &
PID3=$!

echo ""
echo "All 4 GPUs running. Waiting for completion..."
echo "  GPU0 (PID ${PID0}): controls + split sweep"
echo "  GPU1 (PID ${PID1}): bridge dim + competition tech"
echo "  GPU2 (PID ${PID2}): MLP + depth + RoPE"
echo "  GPU3 (PID ${PID3}): combo stacking"
echo ""

wait $PID0; echo "GPU0 DONE — $(date)"
wait $PID1; echo "GPU1 DONE — $(date)"
wait $PID2; echo "GPU2 DONE — $(date)"
wait $PID3; echo "GPU3 DONE — $(date)"

# Merge all GPU results into one summary
SUMMARY="${RESULTS_DIR}/ablation_summary_${TS}.tsv"
echo -e "arm\tparams\traw_bpb\tstep_ms" > "${SUMMARY}"
for g in 0 1 2 3; do
    tail -n +2 "${RESULTS_DIR}/gpu${g}_${TS}.tsv" >> "${SUMMARY}"
done

echo ""
echo "================================================================"
echo "  SPLITHEAD ABLATION COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
