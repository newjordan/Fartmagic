#!/bin/bash
set -euo pipefail
# ================================================================
#  BW13_TapOff_Anchor_GPTQ_2k — Unified Ablation Sequence
#
#  Goal: optimize crawler interactions on the new 4x-friendly stack.
#  All gate arms run 2000 steps.
#
#  Lane A (WINDOW / retrain-required):
#    - architecture/state interaction changes requiring fresh training.
#  Lane B (POST_WINDOW / sequential):
#    - quantization-policy tests on frozen control checkpoint.
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-01_BW13_TapOff_Anchor_GPTQ_2k/run_ablation_sequence.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${REPO_ROOT}/crawler/2026-04-01_BW12_Interaction_2k/train_gpt_ablate.py"
RUN_TAG="BW13INT"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_s${SEED}_${TS}.tsv"

TORCHRUN=$(command -v torchrun 2>/dev/null || echo "python3 -m torch.distributed.run")

# Shared gate baseline: tap-off crawler stack, 2k steps.
BASE_ENV=(
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
    NUM_FLAT_LAYERS=5
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=1
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    GPTQ_CAL_SAMPLES=128
    GPTQ_CAL_SEQ_LEN=2048
    NPROC_PER_NODE="${NPROC}"
)

CONTROL_INT6=""
CONTROL_CKPT="${RESULTS_DIR}/${RUN_TAG}-00_control_s${SEED}_${TS}.final_model.pt"

# summary header
{
    echo -e "lane\tarm\tdesc\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tgptq_layers\tgptq_cal_sec\tdelta_vs_control\tlog"
} > "${SUMMARY}"

calc_delta() {
    local value="$1"
    if [[ -z "${CONTROL_INT6}" || -z "${value}" || "${value}" == "?" ]]; then
        echo "?"
        return
    fi
    python3 - <<PY
c = float("${CONTROL_INT6}")
v = float("${value}")
d = v - c
sign = "+" if d >= 0 else ""
print(f"{sign}{d:.6f}")
PY
}

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

run_arm() {
    local lane="$1"; shift
    local arm="$1"; shift
    local desc="$1"; shift
    local extra_env=("$@")

    local log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ${lane} ${arm}: ${desc}"
    echo "  log: ${log}"
    echo "----------------------------------------------------------------"

    env "${BASE_ENV[@]}" "${extra_env[@]}" \
      ${TORCHRUN} --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
      2>&1 | tee "${log}"

    local raw int6 step_ms bytes gptq_layers gptq_cal_sec delta
    raw=$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    int6=$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    step_ms=$(extract_metric 'step_avg:\K[0-9.]+' "${log}")
    bytes=$(extract_metric 'Total submission size int6\+zstd: \K[0-9]+' "${log}")
    gptq_layers=$(extract_metric 'gptq_quantize: \K[0-9]+' "${log}")
    gptq_cal_sec=$(extract_metric 'gptq:(?:loop-aware )?calibrated [0-9]+ layers in \K[0-9.]+' "${log}")

    if [[ -z "${raw}" ]]; then raw="?"; fi
    if [[ -z "${int6}" ]]; then int6="?"; fi
    if [[ -z "${step_ms}" ]]; then step_ms="?"; fi
    if [[ -z "${bytes}" ]]; then bytes="?"; fi
    if [[ -z "${gptq_layers}" ]]; then gptq_layers="0"; fi
    if [[ -z "${gptq_cal_sec}" ]]; then gptq_cal_sec="-"; fi

    if [[ "${arm}" == "${RUN_TAG}-00" ]]; then
        CONTROL_INT6="${int6}"
        if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
            cp -f "${REPO_ROOT}/final_model.pt" "${CONTROL_CKPT}"
            echo "  control checkpoint: ${CONTROL_CKPT}"
        else
            echo "  WARNING: final_model.pt missing after control arm"
        fi
    fi

    delta=$(calc_delta "${int6}")
    echo -e "${lane}\t${arm}\t${desc}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${gptq_layers}\t${gptq_cal_sec}\t${delta}\t${log}" >> "${SUMMARY}"

    echo "  ${arm}: raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} gptq_layers=${gptq_layers} gptq_cal_sec=${gptq_cal_sec} delta_vs_ctrl=${delta}"
}

run_post_window_arm() {
    local arm="$1"; shift
    local desc="$1"; shift
    if [[ ! -f "${CONTROL_CKPT}" ]]; then
        echo "ERROR: missing control checkpoint ${CONTROL_CKPT}"
        exit 1
    fi
    run_arm "POST_WINDOW" "${arm}" "${desc}" \
        SKIP_TRAIN=1 \
        INIT_MODEL_PATH="${CONTROL_CKPT}" \
        "$@"
}

# ----------------------------------------------------------------
# Lane A: 2k retrain-required interaction tests
# ----------------------------------------------------------------
run_arm "WINDOW" "${RUN_TAG}-00" "control (tap-off Nightcrawler, naive int6)"

run_arm "WINDOW" "${RUN_TAG}-01" "tap-off + anchor dim=32" \
    ANCHOR_DIM=32

run_arm "WINDOW" "${RUN_TAG}-02" "tap-off + anchor dim=64" \
    ANCHOR_DIM=64

# ----------------------------------------------------------------
# Lane B: sequential post-window quant tests (no retrain)
# ----------------------------------------------------------------
run_post_window_arm "${RUN_TAG}-Q0" "naive int6 on frozen control checkpoint" \
    SKIP_GPTQ=1 \
    LOOP_AWARE_GPTQ=0

run_post_window_arm "${RUN_TAG}-Q1" "standard GPTQ (128x2048) on frozen control checkpoint" \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=0 \
    GPTQ_CAL_SAMPLES=128 \
    GPTQ_CAL_SEQ_LEN=2048

run_post_window_arm "${RUN_TAG}-Q1L" "standard GPTQ-lite (64x1024) on frozen control checkpoint" \
    SKIP_GPTQ=0 \
    LOOP_AWARE_GPTQ=0 \
    GPTQ_CAL_SAMPLES=64 \
    GPTQ_CAL_SEQ_LEN=1024

# ----------------------------------------------------------------
# Full-run queue (do NOT auto-run here)
# ----------------------------------------------------------------
cat <<TXT

================================================================
BW13 sequence complete.
summary: ${SUMMARY}

Promotion policy (full 600s 8xH100 required):
  1) Any WINDOW arm with delta_vs_control <= -0.0008
  2) Any POST_WINDOW arm with delta_vs_control <= -0.0008,
     then rerun that quant policy as a full training run (not SKIP_TRAIN)

Classification:
  - WINDOW arms: must retrain (architecture/state interactions)
  - POST_WINDOW arms: sequential after one training window (quant-only interactions)
================================================================
TXT

column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
