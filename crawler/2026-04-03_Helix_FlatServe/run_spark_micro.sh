#!/bin/bash
set -euo pipefail
# Helix FlatServe �� DGX Spark Micro Tests
# The flat layers serve the crawler. Test modifications that make
# flat layer output more denoísable by the crawler's shared weights.
#
# Usage:
#   cd ~/parameter-golf-lab && bash crawler/2026-04-03_Helix_FlatServe/run_spark_micro.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/flatserve_summary_${TS}.tsv"

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
    NUM_CRAWLER_LAYERS=1
    CRAWLER_MLP_MULT=4.0
    MODEL_DIM=256
    NUM_HEADS=4
    NUM_KV_HEADS=2
    NUM_FLAT_LAYERS=5
    CRAWLER_LOOPS=1
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
    TRAIN_LOG_EVERY=50
    VAL_LOSS_EVERY=200
    # Helix ON for all FlatServe tests (dim=64, our confirmed sweet spot at micro)
    HELIX=1
    HELIX_DIM=64
    HELIX_STRIDE=1
)

run_arm() {
    local tag="$1"; shift
    local logfile="${RESULTS_DIR}/${tag}_s${SEED}_${TS}.log"
    echo ""
    echo "================================================================"
    echo "  ARM: ${tag} — $(date)"
    echo "================================================================"
    env "${MICRO_ENV[@]}" "$@" \
        python "${TRAIN_PY}" 2>&1 | tee "${logfile}"
    local bpb=$(grep -oP 'step:200/200 val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local params=$(grep -oP 'model_params:\K[0-9]+' "${logfile}" 2>/dev/null || echo "?")
    echo -e "${tag}\t${params}\t${bpb}\t${step_ms}" >> "${SUMMARY}"
    echo "  >>> ${tag}: bpb=${bpb} step_ms=${step_ms} params=${params}"
}

echo -e "arm\tparams\traw_bpb\tstep_ms" > "${SUMMARY}"

echo ""
echo "================================================================"
echo "  HELIX FLATSERVE — Flat layers serve the crawler"
echo "  All arms: Helix ON (dim=64, stride=1)"
echo "  200 steps, dim=256, seq=512"
echo "================================================================"

# ----------------------------------------------------------------
# CONTROL: Helix with standard flat layers
# ----------------------------------------------------------------

echo ""
echo "==== CONTROL ===="

run_arm "F0_helix_standard" \
    FLAT_RESIDUAL_SCALE=1.0 FLAT_NOISE_STD=0.0 FLAT_SKIP_TO_CRAWL=0

# ----------------------------------------------------------------
# RESIDUAL SCALING: Smaller flat updates = smoother for crawler
# ----------------------------------------------------------------

echo ""
echo "==== RESIDUAL SCALING — Many small steps ===="

run_arm "F1_scale_0.5" \
    FLAT_RESIDUAL_SCALE=0.5 FLAT_NOISE_STD=0.0 FLAT_SKIP_TO_CRAWL=0

run_arm "F2_scale_0.7" \
    FLAT_RESIDUAL_SCALE=0.7 FLAT_NOISE_STD=0.0 FLAT_SKIP_TO_CRAWL=0

run_arm "F3_scale_0.85" \
    FLAT_RESIDUAL_SCALE=0.85 FLAT_NOISE_STD=0.0 FLAT_SKIP_TO_CRAWL=0

# ----------------------------------------------------------------
# NOISE INJECTION: Force crawler to learn robust denoising
# ----------------------------------------------------------------

echo ""
echo "==== NOISE INJECTION — Crawler learns to denoise ===="

run_arm "F4_noise_0.01" \
    FLAT_RESIDUAL_SCALE=1.0 FLAT_NOISE_STD=0.01 FLAT_SKIP_TO_CRAWL=0

run_arm "F5_noise_0.05" \
    FLAT_RESIDUAL_SCALE=1.0 FLAT_NOISE_STD=0.05 FLAT_SKIP_TO_CRAWL=0

run_arm "F6_noise_0.1" \
    FLAT_RESIDUAL_SCALE=1.0 FLAT_NOISE_STD=0.1 FLAT_SKIP_TO_CRAWL=0

# ----------------------------------------------------------------
# SKIP-TO-CRAWL: Route U-Net skips through crawler
# ----------------------------------------------------------------

echo ""
echo "==== SKIP-TO-CRAWL — All info passes through crawler ===="

run_arm "F7_skip_to_crawl" \
    FLAT_RESIDUAL_SCALE=1.0 FLAT_NOISE_STD=0.0 FLAT_SKIP_TO_CRAWL=1

# ----------------------------------------------------------------
# COMBOS: Best of each + stacking
# ----------------------------------------------------------------

echo ""
echo "==== COMBOS ===="

# Scale + noise (crawler sees small, noisy steps — maximum denoising pressure)
run_arm "F8_scale0.7_noise0.05" \
    FLAT_RESIDUAL_SCALE=0.7 FLAT_NOISE_STD=0.05 FLAT_SKIP_TO_CRAWL=0

# Scale + skip-to-crawl
run_arm "F9_scale0.7_skip" \
    FLAT_RESIDUAL_SCALE=0.7 FLAT_NOISE_STD=0.0 FLAT_SKIP_TO_CRAWL=1

# All three
run_arm "F10_all" \
    FLAT_RESIDUAL_SCALE=0.7 FLAT_NOISE_STD=0.05 FLAT_SKIP_TO_CRAWL=1

# ----------------------------------------------------------------
# ABLATION: Same modifications WITHOUT helix (do they help alone?)
# ----------------------------------------------------------------

echo ""
echo "==== ABLATION — Same mods without helix ===="

run_arm "F11_no_helix_ctrl" \
    HELIX=0 FLAT_RESIDUAL_SCALE=1.0 FLAT_NOISE_STD=0.0 FLAT_SKIP_TO_CRAWL=0

run_arm "F12_no_helix_scale0.7" \
    HELIX=0 FLAT_RESIDUAL_SCALE=0.7 FLAT_NOISE_STD=0.0 FLAT_SKIP_TO_CRAWL=0

run_arm "F13_no_helix_noise0.05" \
    HELIX=0 FLAT_RESIDUAL_SCALE=1.0 FLAT_NOISE_STD=0.05 FLAT_SKIP_TO_CRAWL=0

echo ""
echo "================================================================"
echo "  FLATSERVE COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
