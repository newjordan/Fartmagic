#!/bin/bash
set -euo pipefail
# ================================================================
# BW19_CrawlerSystem_2k
#
# One-command crawler interaction matrix on stable 9F floor:
#   1) QUICK WINDOW stage: crawler-system interaction sweep
#   2) FULL WINDOW stage: replay top QUICK deltas at 2k steps
#   3) CALIBRATION stage: post-window quant/calibration on best FULL ckpt
#
# Big swings first, small knobs last.
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
RUN_TAG="BW19CS"
TS="$(date +%Y%m%d_%H%M%S)"

# Stage toggles
RUN_QUICK="${RUN_QUICK:-1}"
RUN_FULL="${RUN_FULL:-1}"
RUN_CALIB="${RUN_CALIB:-1}"
RUN_LOOP_AWARE_GPTQ="${RUN_LOOP_AWARE_GPTQ:-0}"
FULL_TOPK="${FULL_TOPK:-6}"

# QUICK stage defaults (efficient broad screen)
QUICK_ITERATIONS="${QUICK_ITERATIONS:-1200}"
QUICK_MAX_WALLCLOCK_SECONDS="${QUICK_MAX_WALLCLOCK_SECONDS:-2400}"
QUICK_TRAIN_BATCH_TOKENS="${QUICK_TRAIN_BATCH_TOKENS:-786432}"
QUICK_VAL_BATCH_SIZE="${QUICK_VAL_BATCH_SIZE:-524288}"
QUICK_EVAL_STRIDE="${QUICK_EVAL_STRIDE:-0}"

# FULL stage defaults (2k signal confirm)
FULL_ITERATIONS="${FULL_ITERATIONS:-2000}"
FULL_MAX_WALLCLOCK_SECONDS="${FULL_MAX_WALLCLOCK_SECONDS:-3600}"
FULL_TRAIN_BATCH_TOKENS="${FULL_TRAIN_BATCH_TOKENS:-786432}"
FULL_VAL_BATCH_SIZE="${FULL_VAL_BATCH_SIZE:-524288}"
FULL_EVAL_STRIDE="${FULL_EVAL_STRIDE:-0}"

# 9F crawler-system baseline defaults
BASE_CRAWLER_QUANT_INT8="${BASE_CRAWLER_QUANT_INT8:-0}"
BASE_MATRIX_LR="${BASE_MATRIX_LR:-0.03}"
BASE_EMBED_LR="${BASE_EMBED_LR:-0.035}"

DEFAULT_TRAIN_PY_A="${REPO_ROOT}/records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/train_gpt.py"
DEFAULT_TRAIN_PY_B="${REPO_ROOT}/experiments/shroud_nightcrawler/train_gpt.py"
TRAIN_PY="${TRAIN_PY:-}"
if [[ -z "${TRAIN_PY}" ]]; then
    if [[ -f "${DEFAULT_TRAIN_PY_A}" ]]; then
        TRAIN_PY="${DEFAULT_TRAIN_PY_A}"
    elif [[ -f "${DEFAULT_TRAIN_PY_B}" ]]; then
        TRAIN_PY="${DEFAULT_TRAIN_PY_B}"
    else
        echo "ERROR: no compatible train_gpt.py found (checked ${DEFAULT_TRAIN_PY_A} and ${DEFAULT_TRAIN_PY_B})" >&2
        exit 1
    fi
fi

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_s${SEED}_${TS}.tsv"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

BASE_ENV=(
    SEED="${SEED}"
    WARMDOWN_ITERS=2000
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    ROPE_DIMS=16
    SWA_EVERY=50
    SWA_ENABLED=1
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR="${BASE_MATRIX_LR}"
    EMBED_LR="${BASE_EMBED_LR}"
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_ENABLED="${COMPILE_ENABLED:-0}"
    COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"
    DDP_FIND_UNUSED_PARAMETERS=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=9
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8="${BASE_CRAWLER_QUANT_INT8}"
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    MLP_ACT=relu_sq
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_MLP_CHOKE_SHAPE=flat
    CRAWLER_MLP_CHOKE_GROUPS=8
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

QUICK_ENV=(
    ITERATIONS="${QUICK_ITERATIONS}"
    MAX_WALLCLOCK_SECONDS="${QUICK_MAX_WALLCLOCK_SECONDS}"
    TRAIN_BATCH_TOKENS="${QUICK_TRAIN_BATCH_TOKENS}"
    VAL_BATCH_SIZE="${QUICK_VAL_BATCH_SIZE}"
    EVAL_STRIDE="${QUICK_EVAL_STRIDE}"
)

FULL_ENV=(
    ITERATIONS="${FULL_ITERATIONS}"
    MAX_WALLCLOCK_SECONDS="${FULL_MAX_WALLCLOCK_SECONDS}"
    TRAIN_BATCH_TOKENS="${FULL_TRAIN_BATCH_TOKENS}"
    VAL_BATCH_SIZE="${FULL_VAL_BATCH_SIZE}"
    EVAL_STRIDE="${FULL_EVAL_STRIDE}"
)

# QUICK promotion ranking:
# - pareto_int6_bytes: keep non-dominated fronts on (int6_sw_bpb, bytes), then sort by int6 then bytes
# - int6_only: legacy sort by int6_sw_bpb only
QUICK_PICK_MODE="${QUICK_PICK_MODE:-pareto_int6_bytes}"
NO_RERUN_IF_SEEN="${NO_RERUN_IF_SEEN:-1}"
SEEN_SUMMARY_GLOB="${SEEN_SUMMARY_GLOB:-${RESULTS_DIR}/summary_s${SEED}_*.tsv}"

declare -A CONTROL_INT6=()
declare -A CONTROL_BYTES=()
declare -A RUN_INT6=()
declare -A RUN_CKPT=()
declare -A RUN_LOG=()
declare -A RUN_EXIT=()
declare -A SEEN_SOURCE_ARM=()
declare -A SEEN_DESC=()
declare -A SEEN_MUST_RETRAIN=()
declare -A SEEN_SOURCE_CKPT=()
declare -A SEEN_EXIT=()
declare -A SEEN_PARAMS=()
declare -A SEEN_RAW=()
declare -A SEEN_INT6=()
declare -A SEEN_STEP_MS=()
declare -A SEEN_BYTES=()
declare -A SEEN_BYTES_MB=()
declare -A SEEN_SIZE_PER_BPB_MB=()
declare -A SEEN_BPB_X_MB=()
declare -A SEEN_GPTQ_LAYERS=()
declare -A SEEN_GPTQ_CAL_SEC=()
declare -A SEEN_DELTA=()
declare -A SEEN_DELTA_BYTES=()
declare -A SEEN_LOG=()
declare -A SEEN_CKPT=()
declare -A SEEN_SUMMARY_FILE=()

BEST_FULL_ARM=""
BEST_FULL_SOURCE=""
BEST_FULL_INT6=""
BEST_FULL_CKPT=""

{
    echo -e "stage\tlane\tarm\tsource_arm\tdesc\tmust_retrain\tsource_ckpt\texit_code\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tbytes_mb\tsize_per_bpb_mb\tbpb_x_mb\tgptq_layers\tgptq_cal_sec\tdelta_vs_control\tdelta_bytes_vs_control\tlog\tckpt"
} > "${SUMMARY}"

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

is_numeric() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

calc_delta() {
    local control="$1"
    local value="$2"
    if ! is_numeric "${control}" || ! is_numeric "${value}"; then
        echo "?"
        return
    fi
    python3 - "${control}" "${value}" <<'PY'
import sys
c = float(sys.argv[1])
v = float(sys.argv[2])
d = v - c
sign = "+" if d >= 0 else ""
print(f"{sign}{d:.6f}")
PY
}

calc_delta_int() {
    local control="$1"
    local value="$2"
    if ! is_numeric "${control}" || ! is_numeric "${value}"; then
        echo "?"
        return
    fi
    python3 - "${control}" "${value}" <<'PY'
import sys
c = int(float(sys.argv[1]))
v = int(float(sys.argv[2]))
d = v - c
sign = "+" if d >= 0 else ""
print(f"{sign}{d}")
PY
}

calc_bytes_mb() {
    local bytes="$1"
    if ! is_numeric "${bytes}"; then
        echo "?"
        return
    fi
    python3 - "${bytes}" <<'PY'
import sys
b = float(sys.argv[1])
print(f"{b/1_000_000.0:.3f}")
PY
}

calc_size_per_bpb_mb() {
    local bytes="$1"
    local bpb="$2"
    if ! is_numeric "${bytes}" || ! is_numeric "${bpb}"; then
        echo "?"
        return
    fi
    python3 - "${bytes}" "${bpb}" <<'PY'
import sys
b = float(sys.argv[1]) / 1_000_000.0
q = float(sys.argv[2])
if q <= 0:
    print("?")
else:
    print(f"{b/q:.4f}")
PY
}

calc_bpb_x_mb() {
    local bytes="$1"
    local bpb="$2"
    if ! is_numeric "${bytes}" || ! is_numeric "${bpb}"; then
        echo "?"
        return
    fi
    python3 - "${bytes}" "${bpb}" <<'PY'
import sys
b = float(sys.argv[1]) / 1_000_000.0
q = float(sys.argv[2])
print(f"{q*b:.4f}")
PY
}

load_seen_history() {
    if [[ "${NO_RERUN_IF_SEEN}" != "1" ]]; then
        return 0
    fi
    shopt -s nullglob
    local files=( ${SEEN_SUMMARY_GLOB} )
    shopt -u nullglob
    if (( ${#files[@]} == 0 )); then
        return 0
    fi

    IFS=$'\n' files=($(printf '%s\n' "${files[@]}" | sort))
    local f line_no
    local stage lane arm source_arm desc must_retrain source_ckpt exit_code params raw int6 step_ms bytes bytes_mb size_per_bpb_mb bpb_x_mb gptq_layers gptq_cal_sec delta delta_bytes log ckpt

    for f in "${files[@]}"; do
        [[ "${f}" == "${SUMMARY}" ]] && continue
        [[ -f "${f}" ]] || continue
        line_no=0
        while IFS=$'\t' read -r stage lane arm source_arm desc must_retrain source_ckpt exit_code params raw int6 step_ms bytes bytes_mb size_per_bpb_mb bpb_x_mb gptq_layers gptq_cal_sec delta delta_bytes log ckpt; do
            line_no=$((line_no + 1))
            if (( line_no == 1 )); then
                continue
            fi
            [[ -z "${stage}" || -z "${lane}" || -z "${arm}" ]] && continue
            [[ "${exit_code}" != "0" ]] && continue
            local key="${stage}|${lane}|${arm}"
            SEEN_SOURCE_ARM["${key}"]="${source_arm}"
            SEEN_DESC["${key}"]="${desc}"
            SEEN_MUST_RETRAIN["${key}"]="${must_retrain}"
            SEEN_SOURCE_CKPT["${key}"]="${source_ckpt}"
            SEEN_EXIT["${key}"]="${exit_code}"
            SEEN_PARAMS["${key}"]="${params}"
            SEEN_RAW["${key}"]="${raw}"
            SEEN_INT6["${key}"]="${int6}"
            SEEN_STEP_MS["${key}"]="${step_ms}"
            SEEN_BYTES["${key}"]="${bytes}"
            SEEN_BYTES_MB["${key}"]="${bytes_mb}"
            SEEN_SIZE_PER_BPB_MB["${key}"]="${size_per_bpb_mb}"
            SEEN_BPB_X_MB["${key}"]="${bpb_x_mb}"
            SEEN_GPTQ_LAYERS["${key}"]="${gptq_layers}"
            SEEN_GPTQ_CAL_SEC["${key}"]="${gptq_cal_sec}"
            SEEN_DELTA["${key}"]="${delta}"
            SEEN_DELTA_BYTES["${key}"]="${delta_bytes}"
            SEEN_LOG["${key}"]="${log}"
            SEEN_CKPT["${key}"]="${ckpt}"
            SEEN_SUMMARY_FILE["${key}"]="${f}"
        done < "${f}"
    done
}

is_better() {
    local candidate="$1"
    local incumbent="$2"
    if ! is_numeric "${candidate}"; then
        return 1
    fi
    if ! is_numeric "${incumbent}"; then
        return 0
    fi
    python3 - "${candidate}" "${incumbent}" <<'PY'
import sys
cand = float(sys.argv[1])
best = float(sys.argv[2])
sys.exit(0 if cand < best else 1)
PY
}

arm_desc() {
    case "$1" in
        C00) echo "control: 9F floor + crawler(1x3), tap-off, anchor-off, smear-off" ;;
        C01) echo "cadence extreme low: CRAWLER_LOOPS=1" ;;
        C02) echo "cadence low: CRAWLER_LOOPS=2" ;;
        C03) echo "cadence high: CRAWLER_LOOPS=4" ;;
        C04) echo "crawler depth + split recurrence: NUM_CRAWLER_LAYERS=2, CRAWLER_LOOPS=2" ;;
        C05) echo "crawler depth++: NUM_CRAWLER_LAYERS=3, CRAWLER_LOOPS=2" ;;
        C06) echo "crawler-heavier balance: NUM_FLAT_LAYERS=7, NUM_CRAWLER_LAYERS=2, CRAWLER_LOOPS=3" ;;
        C07) echo "crawler-dominant balance: NUM_FLAT_LAYERS=5, NUM_CRAWLER_LAYERS=3, CRAWLER_LOOPS=3" ;;
        C08) echo "pure crawler probe: NUM_FLAT_LAYERS=0, NUM_CRAWLER_LAYERS=4, CRAWLER_LOOPS=2" ;;
        C09) echo "mirror-lite proxy: loops=2 + anchor16 + tap32(shared) + smear" ;;
        C10) echo "mirror-strong proxy: loops=2 + anchor32 + tap32(loop-specific) + smear" ;;
        C11) echo "mirror-deep proxy: loops=3 + anchor32 + tap32(loop-specific) + smear" ;;
        C12) echo "instruction wide on cadence2: INST_DIM=64" ;;
        C13) echo "crawler choke residual-64 on cadence2" ;;
        C14) echo "QK4 gain on cadence2: QK_GAIN_INIT=4.0" ;;
        C15) echo "rope retune on cadence2: CRAWLER_LOOP_ROPE_SCALES=16,4" ;;
        C16) echo "tap deep-only on cadence2: CRAWLER_TAP_DIM=32, deep" ;;
        C17) echo "tap shallow-only on cadence2: CRAWLER_TAP_DIM=32, shallow" ;;
        *) echo "unknown-arm" ;;
    esac
}

arm_overrides() {
    case "$1" in
        C00)
            cat <<'TXT'
NUM_CRAWLER_LAYERS=1
CRAWLER_LOOPS=3
INST_DIM=32
MODEL_DIM=512
CRAWLER_LOOP_ROPE_SCALES=9,1,1
CRAWLER_LOOP_SMEAR=0
CRAWLER_TAP_DIM=0
CRAWLER_TAP_LOOP_SPECIFIC=1
ANCHOR_DIM=0
CRAWLER_MLP_CHOKE_DIM=0
CRAWLER_MLP_CHOKE_SHAPE=flat
CRAWLER_MLP_MULT=6.0
XSA_LAST_N=11
MATRIX_LR=0.03
EMBED_LR=0.035
FLAT_WEIGHT_SHARE=0
TXT
            ;;
        C01)
            cat <<'TXT'
CRAWLER_LOOPS=1
CRAWLER_LOOP_ROPE_SCALES=9
TXT
            ;;
        C02)
            cat <<'TXT'
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
TXT
            ;;
        C03)
            cat <<'TXT'
CRAWLER_LOOPS=4
CRAWLER_LOOP_ROPE_SCALES=16,8,4,1
TXT
            ;;
        C04)
            cat <<'TXT'
NUM_CRAWLER_LAYERS=2
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
TXT
            ;;
        C05)
            cat <<'TXT'
NUM_CRAWLER_LAYERS=3
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
TXT
            ;;
        C06)
            cat <<'TXT'
NUM_FLAT_LAYERS=7
NUM_CRAWLER_LAYERS=2
CRAWLER_LOOPS=3
CRAWLER_LOOP_ROPE_SCALES=9,3,1
TXT
            ;;
        C07)
            cat <<'TXT'
NUM_FLAT_LAYERS=5
NUM_CRAWLER_LAYERS=3
CRAWLER_LOOPS=3
CRAWLER_LOOP_ROPE_SCALES=9,3,1
TXT
            ;;
        C08)
            cat <<'TXT'
NUM_FLAT_LAYERS=0
NUM_CRAWLER_LAYERS=4
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
CRAWLER_TAP_DIM=0
ANCHOR_DIM=0
CRAWLER_LOOP_SMEAR=0
TXT
            ;;
        C09)
            cat <<'TXT'
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
ANCHOR_DIM=16
CRAWLER_TAP_DIM=32
CRAWLER_TAP_LOOP_SPECIFIC=0
CRAWLER_LOOP_SMEAR=1
TXT
            ;;
        C10)
            cat <<'TXT'
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
ANCHOR_DIM=32
CRAWLER_TAP_DIM=32
CRAWLER_TAP_LOOP_SPECIFIC=1
CRAWLER_LOOP_SMEAR=1
TXT
            ;;
        C11)
            cat <<'TXT'
CRAWLER_LOOPS=3
CRAWLER_LOOP_ROPE_SCALES=9,3,1
ANCHOR_DIM=32
CRAWLER_TAP_DIM=32
CRAWLER_TAP_LOOP_SPECIFIC=1
CRAWLER_LOOP_SMEAR=1
TXT
            ;;
        C12)
            cat <<'TXT'
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
INST_DIM=64
TXT
            ;;
        C13)
            cat <<'TXT'
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
CRAWLER_MLP_CHOKE_DIM=64
CRAWLER_MLP_CHOKE_SHAPE=residual
TXT
            ;;
        C14)
            cat <<'TXT'
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
QK_GAIN_INIT=4.0
TXT
            ;;
        C15)
            cat <<'TXT'
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=16,4
TXT
            ;;
        C16)
            cat <<'TXT'
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
CRAWLER_TAP_DIM=32
CRAWLER_TAP_LOOP_SPECIFIC=1
CRAWLER_TAP_LAYERS=deep
TXT
            ;;
        C17)
            cat <<'TXT'
CRAWLER_LOOPS=2
CRAWLER_LOOP_ROPE_SCALES=9,1
CRAWLER_TAP_DIM=32
CRAWLER_TAP_LOOP_SPECIFIC=1
CRAWLER_TAP_LAYERS=shallow
TXT
            ;;
        *)
            echo "ERROR: unknown source_arm=$1" >&2
            return 2
            ;;
    esac
}

validate_cadence_shape() {
    local loops="$1"
    local scales_csv="$2"
    local n_scales
    if [[ -z "${scales_csv}" ]]; then
        return 0
    fi
    n_scales=$(awk -F',' '{print NF}' <<<"${scales_csv}")
    if (( n_scales < loops )); then
        echo "ERROR: CRAWLER_LOOP_ROPE_SCALES (${scales_csv}) has ${n_scales} entries, but CRAWLER_LOOPS=${loops}" >&2
        return 1
    fi
    return 0
}

run_arm() {
    local stage="$1"; shift
    local lane="$1"; shift
    local arm="$1"; shift
    local source_arm="$1"; shift
    local desc="$1"; shift
    local mode="$1"; shift
    local ctrl_key="$1"; shift
    local is_control="$1"; shift
    local must_retrain="$1"; shift
    local source_ckpt="$1"; shift
    local extra_env=("$@")

    local log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"
    local run_env=("${BASE_ENV[@]}")
    local overrides=()
    local loops="3"
    local scales_csv="9,1,1"
    local exit_code=0
    local seen_key="${stage}|${lane}|${arm}"
    local params raw int6 step_ms bytes bytes_mb size_per_bpb_mb bpb_x_mb gptq_layers gptq_cal_sec delta delta_bytes ckpt

    params="?"
    raw="?"
    int6="?"
    step_ms="?"
    bytes="?"
    bytes_mb="?"
    size_per_bpb_mb="?"
    bpb_x_mb="?"
    gptq_layers="0"
    gptq_cal_sec="-"
    delta_bytes="?"
    ckpt="-"

    if [[ "${NO_RERUN_IF_SEEN}" == "1" && "${SEEN_EXIT[${seen_key}]:-}" == "0" ]]; then
        params="${SEEN_PARAMS[${seen_key}]:-?}"
        raw="${SEEN_RAW[${seen_key}]:-?}"
        int6="${SEEN_INT6[${seen_key}]:-?}"
        step_ms="${SEEN_STEP_MS[${seen_key}]:-?}"
        bytes="${SEEN_BYTES[${seen_key}]:-?}"
        bytes_mb="${SEEN_BYTES_MB[${seen_key}]:-?}"
        size_per_bpb_mb="${SEEN_SIZE_PER_BPB_MB[${seen_key}]:-?}"
        bpb_x_mb="${SEEN_BPB_X_MB[${seen_key}]:-?}"
        gptq_layers="${SEEN_GPTQ_LAYERS[${seen_key}]:-0}"
        gptq_cal_sec="${SEEN_GPTQ_CAL_SEC[${seen_key}]:--}"
        log="${SEEN_LOG[${seen_key}]:-${log}}"
        ckpt="${SEEN_CKPT[${seen_key}]:--}"
        exit_code=0

        if [[ "${bytes_mb}" == "?" ]]; then
            bytes_mb=$(calc_bytes_mb "${bytes}")
        fi
        if [[ "${size_per_bpb_mb}" == "?" ]]; then
            size_per_bpb_mb=$(calc_size_per_bpb_mb "${bytes}" "${int6}")
        fi
        if [[ "${bpb_x_mb}" == "?" ]]; then
            bpb_x_mb=$(calc_bpb_x_mb "${bytes}" "${int6}")
        fi

        if [[ "${is_control}" == "1" ]]; then
            CONTROL_INT6["${ctrl_key}"]="${int6}"
            CONTROL_BYTES["${ctrl_key}"]="${bytes}"
        fi

        delta=$(calc_delta "${CONTROL_INT6[${ctrl_key}]:-}" "${int6}")
        delta_bytes=$(calc_delta_int "${CONTROL_BYTES[${ctrl_key}]:-}" "${bytes}")

        RUN_INT6["${arm}"]="${int6}"
        RUN_CKPT["${arm}"]="${ckpt}"
        RUN_LOG["${arm}"]="${log}"
        RUN_EXIT["${arm}"]="${exit_code}"

        if [[ "${stage}" == "FULL" && "${lane}" == "WINDOW" && "${exit_code}" -eq 0 ]]; then
            if is_better "${int6}" "${BEST_FULL_INT6}"; then
                BEST_FULL_ARM="${arm}"
                BEST_FULL_SOURCE="${source_arm}"
                BEST_FULL_INT6="${int6}"
                BEST_FULL_CKPT="${ckpt}"
            fi
        fi

        echo "SKIP_SEEN_DATA ${stage}/${lane}/${arm}: reusing ${SEEN_SUMMARY_FILE[${seen_key}]:-(unknown summary)}"
        echo -e "${stage}\t${lane}\t${arm}\t${source_arm}\t${desc}\t${must_retrain}\t${source_ckpt}\t${exit_code}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${bytes_mb}\t${size_per_bpb_mb}\t${bpb_x_mb}\t${gptq_layers}\t${gptq_cal_sec}\t${delta}\t${delta_bytes}\t${log}\t${ckpt}" >> "${SUMMARY}"
        echo "  ${arm}: exit=${exit_code} raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} bytes_mb=${bytes_mb} size_per_bpb_mb=${size_per_bpb_mb} bpb_x_mb=${bpb_x_mb} delta_vs_ctrl=${delta} delta_bytes_vs_ctrl=${delta_bytes}"
        return 0
    fi

    case "${mode}" in
        QUICK) run_env+=("${QUICK_ENV[@]}") ;;
        FULL|CALIB) run_env+=("${FULL_ENV[@]}") ;;
        *)
            echo "ERROR: unknown mode=${mode}" | tee "${log}" >&2
            exit_code=98
            ;;
    esac

    if [[ "${exit_code}" -eq 0 ]]; then
        if ! mapfile -t overrides < <(arm_overrides "${source_arm}" | sed '/^$/d'); then
            echo "ERROR: override resolution failed for source_arm=${source_arm}" | tee -a "${log}" >&2
            exit_code=97
            overrides=()
        fi
    fi

    if [[ "${exit_code}" -eq 0 ]]; then
        for kv in "${overrides[@]}" "${extra_env[@]}"; do
            case "${kv}" in
                CRAWLER_LOOPS=*)
                    loops="${kv#*=}"
                    ;;
                CRAWLER_LOOP_ROPE_SCALES=*)
                    scales_csv="${kv#*=}"
                    ;;
            esac
        done
        if ! validate_cadence_shape "${loops}" "${scales_csv}" >> "${log}" 2>&1; then
            echo "ERROR: cadence validation failed; skipping run for ${arm}" | tee -a "${log}" >&2
            exit_code=96
        fi
    fi

    run_env+=("${overrides[@]}")
    run_env+=("${extra_env[@]}")

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ${stage} ${lane} ${arm}: ${desc}"
    echo "  source_arm: ${source_arm}"
    echo "  must_retrain: ${must_retrain}"
    echo "  source_ckpt: ${source_ckpt}"
    echo "  log: ${log}"
    echo "----------------------------------------------------------------"

    if [[ "${exit_code}" -eq 0 ]]; then
        set +e
        env "${run_env[@]}" \
          "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
          2>&1 | tee "${log}"
        exit_code=${PIPESTATUS[0]}
        set -e
    else
        echo "SKIP_EXECUTION: pre-run validation failed for ${arm}" >> "${log}"
    fi

    params=$(extract_metric 'model_params:\K[0-9]+' "${log}")
    raw=$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    int6=$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    if [[ -z "${int6}" ]]; then
        int6=$(extract_metric 'final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    fi
    step_ms=$(extract_metric 'step_avg:\K[0-9.]+' "${log}")
    bytes=$(extract_metric 'Total submission size int6\+(?:zstd|zlib): \K[0-9]+' "${log}")
    gptq_layers=$(extract_metric 'gptq_quantize: \K[0-9]+' "${log}")
    gptq_cal_sec=$(extract_metric 'gptq:(?:loop-aware )?calibrated [0-9]+ layers in \K[0-9.]+' "${log}")

    if [[ -z "${params}" ]]; then params="?"; fi
    if [[ -z "${raw}" ]]; then raw="?"; fi
    if [[ -z "${int6}" ]]; then int6="?"; fi
    if [[ -z "${step_ms}" ]]; then step_ms="?"; fi
    if [[ -z "${bytes}" ]]; then bytes="?"; fi
    if [[ -z "${gptq_layers}" ]]; then gptq_layers="0"; fi
    if [[ -z "${gptq_cal_sec}" ]]; then gptq_cal_sec="-"; fi

    if [[ "${is_control}" == "1" ]]; then
        CONTROL_INT6["${ctrl_key}"]="${int6}"
        CONTROL_BYTES["${ctrl_key}"]="${bytes}"
    fi

    delta=$(calc_delta "${CONTROL_INT6[${ctrl_key}]:-}" "${int6}")
    delta_bytes=$(calc_delta_int "${CONTROL_BYTES[${ctrl_key}]:-}" "${bytes}")
    bytes_mb=$(calc_bytes_mb "${bytes}")
    size_per_bpb_mb=$(calc_size_per_bpb_mb "${bytes}" "${int6}")
    bpb_x_mb=$(calc_bpb_x_mb "${bytes}" "${int6}")

    if [[ "${lane}" == "WINDOW" && "${exit_code}" -eq 0 && -f "${REPO_ROOT}/final_model.pt" ]]; then
        ckpt="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.final_model.pt"
        cp -f "${REPO_ROOT}/final_model.pt" "${ckpt}"
    fi

    RUN_INT6["${arm}"]="${int6}"
    RUN_CKPT["${arm}"]="${ckpt}"
    RUN_LOG["${arm}"]="${log}"
    RUN_EXIT["${arm}"]="${exit_code}"

    if [[ "${stage}" == "FULL" && "${lane}" == "WINDOW" && "${exit_code}" -eq 0 ]]; then
        if is_better "${int6}" "${BEST_FULL_INT6}"; then
            BEST_FULL_ARM="${arm}"
            BEST_FULL_SOURCE="${source_arm}"
            BEST_FULL_INT6="${int6}"
            BEST_FULL_CKPT="${ckpt}"
        fi
    fi

    echo -e "${stage}\t${lane}\t${arm}\t${source_arm}\t${desc}\t${must_retrain}\t${source_ckpt}\t${exit_code}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${bytes_mb}\t${size_per_bpb_mb}\t${bpb_x_mb}\t${gptq_layers}\t${gptq_cal_sec}\t${delta}\t${delta_bytes}\t${log}\t${ckpt}" >> "${SUMMARY}"

    echo "  ${arm}: exit=${exit_code} raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} bytes_mb=${bytes_mb} size_per_bpb_mb=${size_per_bpb_mb} bpb_x_mb=${bpb_x_mb} delta_vs_ctrl=${delta} delta_bytes_vs_ctrl=${delta_bytes}"
    return 0
}

pick_top_quick_sources() {
    python3 - "${SUMMARY}" "${FULL_TOPK}" "${QUICK_PICK_MODE}" <<'PY'
import csv
import math
import sys

summary = sys.argv[1]
topk = int(sys.argv[2])
mode = sys.argv[3]
rows = []
with open(summary, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row.get("stage") != "QUICK" or row.get("lane") != "WINDOW":
            continue
        src = row.get("source_arm", "")
        if src == "C00":
            continue
        if row.get("exit_code") != "0":
            continue
        try:
            bpb = float(row.get("int6_sw_bpb", "nan"))
            byt = float(row.get("bytes", "nan"))
        except ValueError:
            continue
        if math.isnan(bpb) or math.isnan(byt):
            continue
        rows.append({"src": src, "bpb": bpb, "bytes": byt})

if mode not in {"pareto_int6_bytes", "int6_only"}:
    mode = "pareto_int6_bytes"

if mode == "int6_only":
    rows.sort(key=lambda r: (r["bpb"], r["bytes"]))
else:
    # Pareto front where lower bpb and lower bytes are both better.
    front = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if (b["bpb"] <= a["bpb"] and b["bytes"] <= a["bytes"]) and (b["bpb"] < a["bpb"] or b["bytes"] < a["bytes"]):
                dominated = True
                break
        if not dominated:
            front.append(a)
    rows = sorted(front, key=lambda r: (r["bpb"], r["bytes"]))
seen = set()
picked = []
for item in rows:
    src = item["src"]
    if not src or src in seen:
        continue
    seen.add(src)
    picked.append(src)
    if len(picked) >= topk:
        break

for src in picked:
    print(src)
PY
}

load_seen_history

# ----------------------------------------------------------------
# Arm matrix (crawler-system interactions on stable 9F floor).
# ----------------------------------------------------------------
QUICK_SOURCE_ARMS_CSV="${QUICK_SOURCE_ARMS_CSV:-C00,C01,C02,C03,C04,C05,C06,C07,C08,C09,C10,C11,C12,C13,C14,C15,C16,C17}"
IFS=',' read -r -a QUICK_SOURCE_ARMS <<<"${QUICK_SOURCE_ARMS_CSV}"

TOP_QUICK_SOURCES=()

if [[ "${RUN_QUICK}" == "1" ]]; then
    for src in "${QUICK_SOURCE_ARMS[@]}"; do
        ctrl=0
        if [[ "${src}" == "C00" ]]; then
            ctrl=1
        fi
        run_arm "QUICK" "WINDOW" "${RUN_TAG}-${src}" "${src}" "$(arm_desc "${src}")" "QUICK" "quick" "${ctrl}" 1 "-"
    done

    if mapfile -t TOP_QUICK_SOURCES < <(pick_top_quick_sources); then
        :
    fi
fi

echo ""
echo "Top QUICK sources (excluding control): ${TOP_QUICK_SOURCES[*]:-(none)}"

if [[ "${RUN_FULL}" == "1" ]]; then
    FULL_SOURCE_ARMS=(C00)

    if [[ -n "${FULL_SOURCE_ARMS_CSV:-}" ]]; then
        IFS=',' read -r -a USER_FULL_ARMS <<<"${FULL_SOURCE_ARMS_CSV}"
        for src in "${USER_FULL_ARMS[@]}"; do
            [[ -z "${src}" ]] && continue
            [[ "${src}" == "C00" ]] && continue
            FULL_SOURCE_ARMS+=("${src}")
        done
    else
        for src in "${TOP_QUICK_SOURCES[@]}"; do
            [[ "${src}" == "C00" ]] && continue
            FULL_SOURCE_ARMS+=("${src}")
        done
    fi

    # De-duplicate while preserving order.
    declare -A _seen=()
    FULL_SOURCE_UNIQ=()
    for src in "${FULL_SOURCE_ARMS[@]}"; do
        if [[ -z "${_seen[${src}]:-}" ]]; then
            _seen["${src}"]=1
            FULL_SOURCE_UNIQ+=("${src}")
        fi
    done
    unset _seen

    idx=0
    for src in "${FULL_SOURCE_UNIQ[@]}"; do
        arm="$(printf "BW19F-%02d" "${idx}")"
        ctrl=0
        if [[ "${idx}" -eq 0 ]]; then
            ctrl=1
        fi
        run_arm "FULL" "WINDOW" "${arm}" "${src}" "full replay from ${src}: $(arm_desc "${src}")" "FULL" "full" "${ctrl}" 1 "-"
        idx=$((idx + 1))
    done
fi

if [[ "${RUN_CALIB}" == "1" && "${RUN_FULL}" == "1" ]]; then
    if [[ -z "${BEST_FULL_CKPT}" || ! -f "${BEST_FULL_CKPT}" ]]; then
        echo "WARN: best FULL checkpoint missing; skipping CALIBRATION stage: ${BEST_FULL_CKPT:-(unset)}" >&2
    else
        run_arm "CALIBRATION" "POST_WINDOW" "BW19Q-00" "${BEST_FULL_SOURCE}" \
            "naive int6 on frozen best FULL checkpoint" "CALIB" "calib" 1 0 "${BEST_FULL_CKPT}" \
            SKIP_TRAIN=1 \
            INIT_MODEL_PATH="${BEST_FULL_CKPT}" \
            SKIP_GPTQ=1 \
            LOOP_AWARE_GPTQ=0

        run_arm "CALIBRATION" "POST_WINDOW" "BW19Q-I8" "${BEST_FULL_SOURCE}" \
            "naive int6 with crawler int8 on frozen best FULL checkpoint" "CALIB" "calib" 0 0 "${BEST_FULL_CKPT}" \
            SKIP_TRAIN=1 \
            INIT_MODEL_PATH="${BEST_FULL_CKPT}" \
            SKIP_GPTQ=1 \
            LOOP_AWARE_GPTQ=0 \
            CRAWLER_QUANT_INT8=1

        run_arm "CALIBRATION" "POST_WINDOW" "BW19Q-01" "${BEST_FULL_SOURCE}" \
            "standard GPTQ (128x2048) on frozen best FULL checkpoint" "CALIB" "calib" 0 0 "${BEST_FULL_CKPT}" \
            SKIP_TRAIN=1 \
            INIT_MODEL_PATH="${BEST_FULL_CKPT}" \
            SKIP_GPTQ=0 \
            LOOP_AWARE_GPTQ=0 \
            GPTQ_CAL_SAMPLES=128 \
            GPTQ_CAL_SEQ_LEN=2048

        run_arm "CALIBRATION" "POST_WINDOW" "BW19Q-01L" "${BEST_FULL_SOURCE}" \
            "GPTQ-lite (64x1024) on frozen best FULL checkpoint" "CALIB" "calib" 0 0 "${BEST_FULL_CKPT}" \
            SKIP_TRAIN=1 \
            INIT_MODEL_PATH="${BEST_FULL_CKPT}" \
            SKIP_GPTQ=0 \
            LOOP_AWARE_GPTQ=0 \
            GPTQ_CAL_SAMPLES=64 \
            GPTQ_CAL_SEQ_LEN=1024

        run_arm "CALIBRATION" "POST_WINDOW" "BW19Q-01H" "${BEST_FULL_SOURCE}" \
            "GPTQ-heavy (256x2048) on frozen best FULL checkpoint" "CALIB" "calib" 0 0 "${BEST_FULL_CKPT}" \
            SKIP_TRAIN=1 \
            INIT_MODEL_PATH="${BEST_FULL_CKPT}" \
            SKIP_GPTQ=0 \
            LOOP_AWARE_GPTQ=0 \
            GPTQ_CAL_SAMPLES=256 \
            GPTQ_CAL_SEQ_LEN=2048

        if [[ "${RUN_LOOP_AWARE_GPTQ}" == "1" ]]; then
            run_arm "CALIBRATION" "POST_WINDOW" "BW19Q-02" "${BEST_FULL_SOURCE}" \
                "loop-aware GPTQ (128x2048) on frozen best FULL checkpoint" "CALIB" "calib" 0 0 "${BEST_FULL_CKPT}" \
                SKIP_TRAIN=1 \
                INIT_MODEL_PATH="${BEST_FULL_CKPT}" \
                SKIP_GPTQ=0 \
                LOOP_AWARE_GPTQ=1 \
                GPTQ_CAL_SAMPLES=128 \
                GPTQ_CAL_SEQ_LEN=2048
        fi
    fi
fi

cat <<TXT

================================================================
BW19 crawler-system sequence complete.
summary: ${SUMMARY}
train_py: ${TRAIN_PY}

Stage policy:
  - QUICK WINDOW: crawler-interaction screening on 9F floor (must retrain)
  - FULL WINDOW: replay control + top QUICK arms at 2k (must retrain)
  - CALIBRATION POST_WINDOW: quant/calibration-only on best FULL checkpoint

Best FULL WINDOW:
  arm=${BEST_FULL_ARM:-(not run)}
  source_arm=${BEST_FULL_SOURCE:-(not run)}
  int6_sw_bpb=${BEST_FULL_INT6:-(not run)}
  ckpt=${BEST_FULL_CKPT:-(not run)}
================================================================
TXT

column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
