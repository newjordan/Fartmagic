#!/bin/bash
set -euo pipefail
# ================================================================
#  E2E TTT Ablation — 4 arms sequential on 4×GPU, 2000 steps
#
#  Arm 0: Ouroboros control (HELIX=0, loops=3, no TTT)
#  Arm 1: Ouroboros + TTT (HELIX=0, loops=3, TTT_DIM=32)
#  Arm 2: Helix SplitHead control (HELIX=1, cross=8, no TTT)
#  Arm 3: Helix SplitHead + TTT (HELIX=1, cross=8, TTT_DIM=32)
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-06_TTT_ablation/gate.sh
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
TRAIN_PY="${SCRIPT_DIR}/train_gpt_ttt.py"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_s${SEED}_${TS}.tsv"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

# Shared env (common to both architectures)
SHARED_ENV=(
    SEED="${SEED}"
    ITERATIONS=2000
    MAX_WALLCLOCK_SECONDS=3600
    WARMDOWN_ITERS=2000
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    ROPE_DIMS=16
    SWA_EVERY=50
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR=0.03
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_FULLGRAPH=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_CRAWLER_LAYERS=1
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_MLP_CHOKE_SHAPE=flat
    CRAWLER_MLP_CHOKE_GROUPS=8
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    NPROC_PER_NODE="${NPROC}"
)

# BWX 9F base for Ouroboros arms (matches records/.../2026-04-02_Bandit_Wagon_X_9F_8xH100)
OURO_BASE=(
    NUM_FLAT_LAYERS=9
    CRAWLER_LOOPS=3
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_QUANT_INT8=0
)

# BW5 4F base for Helix arms (matches Helix_ab_3 gate)
HELIX_BASE=(
    NUM_FLAT_LAYERS=4
    CRAWLER_LOOPS=3
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_QUANT_INT8=1
    MUON_WD=0.04
    BIGRAM_DIM=128
)

{
    echo -e "arm\tdesc\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_ctrl"
} > "${SUMMARY}"

CTRL_OURO=""
CTRL_HELIX=""
LAST_INT6=""

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

run_arm() {
    local arm="$1"; shift
    local desc="$1"; shift
    local ctrl_ref="$1"; shift
    local extra_env=("$@")

    local log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"

    echo ""
    echo "================================================================"
    echo "  ${arm}: ${desc}"
    echo "  seed=${SEED}  GPUs=${NPROC}  steps=2000"
    echo "  log: ${log}"
    echo "================================================================"
    echo ""

    env "${SHARED_ENV[@]}" "${extra_env[@]}" \
      "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
      2>&1 | tee "${log}"

    local params raw int6 step_ms bytes delta
    params=$(extract_metric 'model_params:\K[0-9]+' "${log}")
    raw=$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    int6=$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    step_ms=$(extract_metric 'step_avg:\K[0-9.]+' "${log}")
    bytes=$(extract_metric 'Total submission size int6\+(?:zstd|zlib|brotli): \K[0-9]+' "${log}")

    [[ -n "${params}" ]]  || params="?"
    [[ -n "${raw}" ]]     || raw="?"
    [[ -n "${int6}" ]]    || int6="?"
    [[ -n "${step_ms}" ]] || step_ms="?"
    [[ -n "${bytes}" ]]   || bytes="?"

    delta=$(calc_delta "${ctrl_ref}" "${int6}")

    echo -e "${arm}\t${desc}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${delta}" >> "${SUMMARY}"
    echo ""
    echo "  >> ${arm}: params=${params} raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} delta=${delta}"
    echo ""

    LAST_INT6="${int6}"
}

# ================================================================
#  ARM 0: Ouroboros control — BWX 9F base (no TTT)
# ================================================================
run_arm "TTT-00_ouro_ctrl" "Ouroboros 9F control (loops=3, no TTT)" "" \
    "${OURO_BASE[@]}" \
    HELIX=0 \
    CRAWLER_CROSS_HEADS=0 \
    TTT_DIM=0
CTRL_OURO="${LAST_INT6}"

# ================================================================
#  ARM 1: Ouroboros + TTT — BWX 9F base
# ================================================================
run_arm "TTT-01_ouro_ttt" "Ouroboros 9F + E2E TTT (loops=3, ttt_dim=32)" "${CTRL_OURO}" \
    "${OURO_BASE[@]}" \
    HELIX=0 \
    CRAWLER_CROSS_HEADS=0 \
    TTT_DIM=32 \
    TTT_LR=0.01 \
    TTT_STEPS=1

# ================================================================
#  ARM 2: Helix SplitHead control — BW5 4F base (no TTT)
# ================================================================
run_arm "TTT-02_helix_ctrl" "Helix SplitHead 4F control (cross=8, no TTT)" "" \
    "${HELIX_BASE[@]}" \
    HELIX=1 \
    HELIX_DIM=384 \
    HELIX_STRIDE=1 \
    HELIX_CROSS_ATTN=0 \
    CRAWLER_CROSS_HEADS=8 \
    CRAWLER_LOOPS=1 \
    CRAWLER_LOOP_ROPE_SCALES=9 \
    CRAWLER_V0_RESIDUAL=0 \
    TTT_DIM=0
CTRL_HELIX="${LAST_INT6}"

# ================================================================
#  ARM 3: Helix SplitHead + TTT — BW5 4F base
# ================================================================
run_arm "TTT-03_helix_ttt" "Helix SplitHead 4F + E2E TTT (cross=8, ttt_dim=32)" "${CTRL_HELIX}" \
    "${HELIX_BASE[@]}" \
    HELIX=1 \
    HELIX_DIM=384 \
    HELIX_STRIDE=1 \
    HELIX_CROSS_ATTN=0 \
    CRAWLER_CROSS_HEADS=8 \
    CRAWLER_LOOPS=1 \
    CRAWLER_LOOP_ROPE_SCALES=9 \
    CRAWLER_V0_RESIDUAL=0 \
    TTT_DIM=32 \
    TTT_LR=0.01 \
    TTT_STEPS=1

# ================================================================
#  Summary
# ================================================================
echo ""
echo "================================================================"
echo "  E2E TTT Ablation — Results Summary"
echo "  seed=${SEED}  GPUs=${NPROC}  steps=2000"
echo "================================================================"
echo ""
column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
echo ""
echo "  Ouroboros control: ${CTRL_OURO}"
echo "  Helix control:    ${CTRL_HELIX}"
echo ""
echo "  Results saved: ${SUMMARY}"
echo "================================================================"
