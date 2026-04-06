#!/bin/bash
set -euo pipefail
# ================================================================
#  Helix_ab_3 — Gate: Helix SplitHead Architecture Isolation
#
#  2-arm sequential ablation on 4×GPU, 2000 steps each.
#  ONE variable: Helix SplitHead co-firing vs sequential crawler loops.
#  All hyperparameters (including MUON_WD=0.04) held at BW5 defaults.
#
#  Arm 0 (CTRL): BW5 standard — HELIX=0, CRAWLER_LOOPS=3
#  Arm 1 (HELIX): SplitHead — HELIX=1, dim=384, stride=1, cross_heads=8, loops=1
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-06_Helix_ab_3/gate.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_s${SEED}_${TS}.tsv"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

# ----------------------------------------------------------------
# BW5 baseline config — shared across both arms
# ----------------------------------------------------------------
BASE_ENV=(
    SEED="${SEED}"
    ITERATIONS=2000
    MAX_WALLCLOCK_SECONDS=3600
    WARMDOWN_ITERS=2000
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    BIGRAM_DIM=128
    ROPE_DIMS=16
    SWA_EVERY=50
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR=0.03
    MUON_WD=0.04
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_FULLGRAPH=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=4
    NUM_CRAWLER_LAYERS=1
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=1
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    NITRUST_ENABLE=0
    NITRUST_STRICT=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    NPROC_PER_NODE="${NPROC}"
)

# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------
{
    echo -e "arm\tdesc\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_ctrl"
} > "${SUMMARY}"

CTRL_INT6=""
LAST_INT6=""

# ----------------------------------------------------------------
# Metric extraction
# ----------------------------------------------------------------
extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

calc_delta() {
    local control="$1"
    local value="$2"
    if [[ -z "${control}" || -z "${value}" || "${control}" == "?" || "${value}" == "?" ]]; then
        echo "?"
        return
    fi
    python3 - <<PY
c = float("${control}")
v = float("${value}")
d = v - c
sign = "+" if d >= 0 else ""
print(f"{sign}{d:.8f}")
PY
}

# ----------------------------------------------------------------
# run_arm — train one arm and extract metrics
# ----------------------------------------------------------------
run_arm() {
    local arm="$1"; shift
    local desc="$1"; shift
    local extra_env=("$@")

    local log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"

    echo ""
    echo "================================================================"
    echo "  ${arm}: ${desc}"
    echo "  seed=${SEED}  GPUs=${NPROC}  steps=2000"
    echo "  log: ${log}"
    echo "================================================================"
    echo ""

    env "${BASE_ENV[@]}" "${extra_env[@]}" \
      "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
      2>&1 | tee "${log}"

    local params raw int6 step_ms bytes delta
    params=$(extract_metric 'model_params:\K[0-9]+' "${log}")
    raw=$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    int6=$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    step_ms=$(extract_metric 'step_avg:\K[0-9.]+' "${log}")
    bytes=$(extract_metric 'Total submission size int6\+(?:zstd|zlib): \K[0-9]+' "${log}")

    [[ -n "${params}" ]]  || params="?"
    [[ -n "${raw}" ]]     || raw="?"
    [[ -n "${int6}" ]]    || int6="?"
    [[ -n "${step_ms}" ]] || step_ms="?"
    [[ -n "${bytes}" ]]   || bytes="?"

    delta=$(calc_delta "${CTRL_INT6}" "${int6}")

    echo -e "${arm}\t${desc}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${delta}" >> "${SUMMARY}"

    echo ""
    echo "  >> ${arm}: params=${params} raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} delta=${delta}"
    echo ""

    # Export int6 for control tracking (set global LAST_INT6)
    LAST_INT6="${int6}"
}

# ================================================================
#  ARM 0: CONTROL — BW5 standard (sequential crawler loops)
# ================================================================
run_arm "HAB3-00_ctrl" "BW5 standard (HELIX=0, loops=3, rope=9,1,1)" \
    HELIX=0 \
    CRAWLER_LOOPS=3 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
    CRAWLER_CROSS_HEADS=0

CTRL_INT6="${LAST_INT6}"

echo ""
echo "  Control int6_sw_bpb: ${CTRL_INT6}"
echo ""

# ================================================================
#  ARM 1: HELIX — SplitHead dual-stream co-firing
# ================================================================
run_arm "HAB3-01_helix" "Helix SplitHead (dim=384, stride=1, cross=8, loops=1)" \
    HELIX=1 \
    HELIX_DIM=384 \
    HELIX_STRIDE=1 \
    HELIX_CROSS_ATTN=0 \
    CRAWLER_CROSS_HEADS=8 \
    CRAWLER_LOOPS=1 \
    CRAWLER_LOOP_ROPE_SCALES=9 \
    CRAWLER_V0_RESIDUAL=0

# ================================================================
#  Summary
# ================================================================
echo ""
echo "================================================================"
echo "  Helix_ab_3 Gate — Results Summary"
echo "  seed=${SEED}  GPUs=${NPROC}  steps=2000"
echo "================================================================"
echo ""
column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
echo ""
echo "  Results saved: ${SUMMARY}"
echo "  Gate target: delta < -0.003 int6_sw_bpb"
echo ""
echo "  If PASS → proceed to Helix_ab_4 (WD=0.12)"
echo "  If FAIL → check step_ms, review hypothesis"
echo "================================================================"
