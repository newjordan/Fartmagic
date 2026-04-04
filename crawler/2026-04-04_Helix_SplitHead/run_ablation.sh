#!/bin/bash
set -euo pipefail
# ================================================================
# Helix SplitHead — Master Ablation Suite
#
# Split-head cross-attention: crawler heads attend to flat stream
# + competition tech sweep for crawler-specific benefits
#
# 1×H100, micro config, 200 steps each
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
SUMMARY="${RESULTS_DIR}/ablation_summary_${TS}.tsv"

pip install brotli -q 2>/dev/null || true

# Helix ON, dim=64, stride=1 for all arms (our confirmed base)
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
echo "  HELIX SPLITHEAD — Master Ablation Suite"
echo "  Split-head cross-attention + competition tech sweep"
echo "  200 steps, dim=256, 1×GPU"
echo "================================================================"

# ================================================================
# SECTION 1: CONTROLS
# ================================================================

echo ""
echo "==== SECTION 1: CONTROLS ===="

# S0: Helix standard (no split heads, our baseline)
run_arm "S0_helix_ctrl" \
    CRAWLER_CROSS_HEADS=0

# S1: No helix at all (raw crawler baseline)
run_arm "S1_no_helix" \
    HELIX=0 CRAWLER_CROSS_HEADS=0

# ================================================================
# SECTION 2: SPLIT-HEAD SWEEP — How many heads cross-attend?
# (4 total heads in micro config, 2 KV heads)
# ================================================================

echo ""
echo "==== SECTION 2: SPLIT-HEAD SWEEP ===="

# H1: 1 of 4 heads cross-attend (25% cross)
run_arm "H1_cross1" \
    CRAWLER_CROSS_HEADS=1

# H2: 2 of 4 heads cross-attend (50% cross — the half-and-half)
run_arm "H2_cross2" \
    CRAWLER_CROSS_HEADS=2

# H3: 3 of 4 heads cross-attend (75% cross)
run_arm "H3_cross3" \
    CRAWLER_CROSS_HEADS=3

# H4: ALL 4 heads cross-attend (100% — full cross-attention crawler)
run_arm "H4_cross4_full" \
    CRAWLER_CROSS_HEADS=4

# ================================================================
# SECTION 3: SPLIT-HEAD + BRIDGE DIM
# Best cross-head count from above × dim sweep
# ================================================================

echo ""
echo "==== SECTION 3: SPLIT-HEAD + BRIDGE DIM ===="

# D1: cross=2 + dim=32 (narrow bridge + split)
run_arm "D1_cross2_dim32" \
    CRAWLER_CROSS_HEADS=2 HELIX_DIM=32

# D2: cross=2 + dim=128 (wide bridge + split)
run_arm "D2_cross2_dim128" \
    CRAWLER_CROSS_HEADS=2 HELIX_DIM=128

# D3: cross=4 + dim=32
run_arm "D3_cross4_dim32" \
    CRAWLER_CROSS_HEADS=4 HELIX_DIM=32

# D4: cross=4 + dim=128
run_arm "D4_cross4_dim128" \
    CRAWLER_CROSS_HEADS=4 HELIX_DIM=128

# ================================================================
# SECTION 4: COMPETITION TECH — Crawler-specific benefits
# ================================================================

echo ""
echo "==== SECTION 4: COMPETITION TECH ===="

# W1: Higher weight decay (WD=0.09, field standard — addresses quant gap)
run_arm "W1_wd009" \
    CRAWLER_CROSS_HEADS=2 MUON_WD=0.09

# W2: Even higher (WD=0.12)
run_arm "W2_wd012" \
    CRAWLER_CROSS_HEADS=2 MUON_WD=0.12

# Q1: QK_GAIN=5.0 (field best, PR #1296)
run_arm "Q1_qk5" \
    CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=5.0

# Q2: QK_GAIN=6.0 (push further)
run_arm "Q2_qk6" \
    CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=6.0

# L1: Lower matrix LR (0.02, may help shared weight convergence)
run_arm "L1_lr002" \
    CRAWLER_CROSS_HEADS=2 MATRIX_LR=0.02

# L2: Higher matrix LR (0.04)
run_arm "L2_lr004" \
    CRAWLER_CROSS_HEADS=2 MATRIX_LR=0.04

# ================================================================
# SECTION 5: CRAWLER MLP TUNING
# ================================================================

echo ""
echo "==== SECTION 5: CRAWLER MLP ===="

# M1: Wider crawler MLP (6.0 instead of 4.0)
run_arm "M1_crawl_mlp6" \
    CRAWLER_CROSS_HEADS=2 CRAWLER_MLP_MULT=6.0

# M2: Narrower crawler MLP (2.0)
run_arm "M2_crawl_mlp2" \
    CRAWLER_CROSS_HEADS=2 CRAWLER_MLP_MULT=2.0

# M3: Wider with more flat layers (7F)
run_arm "M3_7f_cross2" \
    CRAWLER_CROSS_HEADS=2 NUM_FLAT_LAYERS=7

# ================================================================
# SECTION 6: DEPTH + SPLIT-HEAD
# ================================================================

echo ""
echo "==== SECTION 6: DEPTH SCALING ===="

# K1: 7F + cross=2 + dim=64
run_arm "K1_7f_cross2_dim64" \
    CRAWLER_CROSS_HEADS=2 NUM_FLAT_LAYERS=7

# K2: 7F + cross=4 (full cross)
run_arm "K2_7f_cross4" \
    CRAWLER_CROSS_HEADS=4 NUM_FLAT_LAYERS=7

# K3: 7F control (no split, just helix)
run_arm "K3_7f_helix_ctrl" \
    CRAWLER_CROSS_HEADS=0 NUM_FLAT_LAYERS=7

# ================================================================
# SECTION 7: ROPE vs NO-ROPE on cross heads
# Already no-RoPE on cross K (position-agnostic) — test WITH RoPE
# ================================================================

echo ""
echo "==== SECTION 7: ROPE VARIATIONS ===="

# R1: cross=2, different crawler RoPE scales
run_arm "R1_rope_1_1_1" \
    CRAWLER_CROSS_HEADS=2 CRAWLER_LOOP_ROPE_SCALES=1,1,1

# R2: cross=2, aggressive RoPE battery
run_arm "R2_rope_16_4_1" \
    CRAWLER_CROSS_HEADS=2 CRAWLER_LOOP_ROPE_SCALES=16,4,1

# ================================================================
# SECTION 8: BEST COMBO STACKING
# ================================================================

echo ""
echo "==== SECTION 8: BEST COMBO CANDIDATES ===="

# B1: cross=2 + QK5 + WD=0.09 (three competition signals)
run_arm "B1_cross2_qk5_wd09" \
    CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=5.0 MUON_WD=0.09

# B2: cross=2 + QK5 + dim=128 + 7F
run_arm "B2_cross2_qk5_dim128_7f" \
    CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=5.0 HELIX_DIM=128 NUM_FLAT_LAYERS=7

# B3: cross=4 (full) + QK5 + WD=0.09 + dim=128
run_arm "B3_full_cross_qk5_wd09_dim128" \
    CRAWLER_CROSS_HEADS=4 QK_GAIN_INIT=5.0 MUON_WD=0.09 HELIX_DIM=128

# B4: kitchen sink — cross=2 + QK5 + WD=0.09 + dim=128 + 7F + crawl_mlp=6
run_arm "B4_kitchen_sink" \
    CRAWLER_CROSS_HEADS=2 QK_GAIN_INIT=5.0 MUON_WD=0.09 HELIX_DIM=128 NUM_FLAT_LAYERS=7 CRAWLER_MLP_MULT=6.0

echo ""
echo "================================================================"
echo "  SPLITHEAD ABLATION COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
