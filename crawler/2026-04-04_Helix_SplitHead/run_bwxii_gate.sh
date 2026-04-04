#!/bin/bash
set -euo pipefail
# BW XII Gate — Helix SplitHead production-scale confirmation
# Side-by-side: control (no helix) vs SplitHead on 4×H100
# 2k steps, full model dim, compile on

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

RESULTS_DIR="${SCRIPT_DIR}/results/bwxii_gate"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/gate_summary_${TS}.tsv"

pip install brotli -q 2>/dev/null || true
rm -rf final_model* *.pt *.ptz

BASE_ENV=(
    SEED="${SEED}"
    ITERATIONS=2000
    MAX_WALLCLOCK_SECONDS=0
    WARMDOWN_ITERS=500
    TRAIN_BATCH_TOKENS=786432
    VAL_BATCH_SIZE=524288
    EVAL_STRIDE=64
    TRAIN_SEQ_LEN=2048
    EVAL_SEQ_LEN=2048
    COMPILE_ENABLED=1
    COMPILE_FULLGRAPH=1
    TORCHDYNAMO_OPTIMIZE_DDP=0
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=5
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=1
    CRAWLER_MLP_MULT=4.0
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
    TRAIN_LOG_EVERY=500
    VAL_LOSS_EVERY=2000
    NPROC_PER_NODE="${NPROC}"
)

run_arm() {
    local tag="$1"; shift
    local logfile="${RESULTS_DIR}/${tag}_s${SEED}_${TS}.log"
    echo ""
    echo "================================================================"
    echo "  ${tag} — $(date)"
    echo "================================================================"
    rm -f final_model* *.pt *.ptz
    env "${BASE_ENV[@]}" "$@" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" 2>&1 | tee "${logfile}"
    local bpb=$(grep -oP 'val_bpb:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local int6=$(grep -oP 'final_int6_sliding_window_exact.*val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local params=$(grep -oP 'model_params:\K[0-9]+' "${logfile}" 2>/dev/null || echo "?")
    echo -e "${tag}\t${params}\t${bpb}\t${int6}\t${step_ms}" >> "${SUMMARY}"
    echo ""
    echo "  >>> ${tag}: bpb=${bpb} int6=${int6} step_ms=${step_ms} params=${params}"
}

echo -e "arm\tparams\traw_bpb\tint6_sw_bpb\tstep_ms" > "${SUMMARY}"

echo ""
echo "================================================================"
echo "  BW XII GATE — Helix SplitHead vs Control"
echo "  2k steps, dim=512, ${NPROC} GPUs, compile=on"
echo "================================================================"

# Control: standard sequential crawler, no helix
run_arm "CTRL_no_helix" \
    HELIX=0 CRAWLER_CROSS_HEADS=0 MUON_WD=0.04

# BW XII: full cross-attention + fat pipe + high WD
run_arm "BWXII_splithead" \
    HELIX=1 HELIX_DIM=192 HELIX_STRIDE=1 CRAWLER_CROSS_HEADS=4 MUON_WD=0.12

echo ""
echo "================================================================"
echo "  BW XII GATE COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
