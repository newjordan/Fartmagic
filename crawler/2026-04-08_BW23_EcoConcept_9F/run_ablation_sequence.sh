#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# BW23_EcoConcept_9F
#
# Mirrored concept ablation matrix:
#   - MODE=gate  : full 2k gate pass
#   - MODE=smoke : DGX Spark smoke pass (same arms, cheaper settings)
#
# Arms:
#   WINDOW: QAT surrogate variants
#   QUANT : INT6_CATS sensitivity-style export policies (post-window)
#
# Usage:
#   SEED=444 NPROC_PER_NODE=8 MODE=gate  bash crawler/2026-04-08_BW23_EcoConcept_9F/run_ablation_sequence.sh
#   SEED=444 NPROC_PER_NODE=4 MODE=smoke bash crawler/2026-04-08_BW23_EcoConcept_9F/run_ablation_sequence.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
MODE="${MODE:-gate}"

case "${MODE}" in
    gate)
        DEFAULT_NPROC=8
        ITERATIONS="${ITERATIONS:-2000}"
        MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3600}"
        TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
        VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
        EVAL_STRIDE="${EVAL_STRIDE:-64}"
        ;;
    smoke)
        DEFAULT_NPROC=4
        ITERATIONS="${ITERATIONS:-12000}"
        MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-240}"
        TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-393216}"
        VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
        EVAL_STRIDE="${EVAL_STRIDE:-2048}"
        ;;
    *)
        echo "ERROR: MODE must be 'gate' or 'smoke', got '${MODE}'" >&2
        exit 1
        ;;
esac

NPROC="${NPROC_PER_NODE:-${DEFAULT_NPROC}}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Tokenizer/data path auto-detection for varied Spark layouts.
if [[ -z "${TOKENIZER_PATH:-}" ]]; then
    for cand in \
        "${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model" \
        "/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_1024_bpe.model" \
        "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model" \
        "/workspace/parameter-golf-lab/data/tokenizers/fineweb_1024_bpe.model"; do
        if [[ -f "${cand}" ]]; then
            TOKENIZER_PATH="${cand}"
            break
        fi
    done
fi

if [[ -z "${DATA_PATH:-}" ]]; then
    for cand in \
        "${REPO_ROOT}/data/datasets/fineweb10B_sp1024" \
        "/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp1024" \
        "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024" \
        "/workspace/parameter-golf-lab/data/datasets/fineweb10B_sp1024"; do
        if [[ -d "${cand}" ]]; then
            DATA_PATH="${cand}"
            break
        fi
    done
fi

if [[ -z "${TOKENIZER_PATH:-}" || ! -f "${TOKENIZER_PATH}" ]]; then
    echo "ERROR: tokenizer not found. Set TOKENIZER_PATH=/abs/path/fineweb_1024_bpe.model" >&2
    exit 1
fi
if [[ -z "${DATA_PATH:-}" || ! -d "${DATA_PATH}" ]]; then
    echo "ERROR: dataset dir not found. Set DATA_PATH=/abs/path/fineweb10B_sp1024" >&2
    exit 1
fi
if ! compgen -G "${DATA_PATH}/fineweb_train_*.bin" >/dev/null; then
    echo "ERROR: no training shards found under ${DATA_PATH} (expected fineweb_train_*.bin)" >&2
    exit 1
fi
if ! compgen -G "${DATA_PATH}/fineweb_val_*.bin" >/dev/null; then
    echo "ERROR: no validation shards found under ${DATA_PATH} (expected fineweb_val_*.bin)" >&2
    exit 1
fi

echo "tokenizer_path=${TOKENIZER_PATH}"
echo "data_path=${DATA_PATH}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_DIR}/summary_${MODE}_s${SEED}_${TS}.tsv"

# Use the active Python env explicitly. Some machines have torchrun shebang
# pinned to /usr/bin/python3 (CPU-only torch), which bypasses conda/venv.
TORCHRUN=("${PYTHON_BIN}" -m torch.distributed.run)

CUDA_INFO="$("${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"ERR: import torch failed: {e}")
    raise SystemExit(2)
print(f"exe={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={int(torch.cuda.is_available())}")
print(f"cuda_devices={torch.cuda.device_count()}")
PY
)"
echo "${CUDA_INFO}"

if ! grep -q "cuda_available=1" <<<"${CUDA_INFO}"; then
    echo "ERROR: CUDA is not available in ${PYTHON_BIN} environment. Activate a GPU torch env first." >&2
    exit 1
fi

CUDA_DEVICES="$(sed -n 's/^cuda_devices=//p' <<<"${CUDA_INFO}")"
if [[ -n "${CUDA_DEVICES}" && "${CUDA_DEVICES}" =~ ^[0-9]+$ ]] && (( NPROC > CUDA_DEVICES )); then
    if [[ "${MODE}" == "smoke" ]]; then
        echo "WARN: NPROC_PER_NODE=${NPROC} > cuda_devices=${CUDA_DEVICES}; auto-setting NPROC_PER_NODE=${CUDA_DEVICES} for smoke mode."
        NPROC="${CUDA_DEVICES}"
    else
        echo "ERROR: NPROC_PER_NODE=${NPROC} but cuda_devices=${CUDA_DEVICES}. Lower NPROC_PER_NODE." >&2
        exit 1
    fi
fi

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
sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
PY
}

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

BASE_ENV=(
    TOKENIZER_PATH="${TOKENIZER_PATH}"
    DATA_PATH="${DATA_PATH}"
    SEED="${SEED}"
    ITERATIONS="${ITERATIONS}"
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}"
    WARMDOWN_ITERS="${ITERATIONS}"
    TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}"
    EVAL_STRIDE="${EVAL_STRIDE}"
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
    NUM_FLAT_LAYERS=9
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=3
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=0
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
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
    NPROC_PER_NODE="${NPROC}"
)

CONTROL_INT6_WINDOW=""
CONTROL_INT6_QUANT=""
BEST_WINDOW_INT6=""
BEST_WINDOW_ARM=""
BEST_WINDOW_CKPT=""

echo -e "phase\tarm\tdesc\tqat_enabled\tqat_surrogate\tint6_cats\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_ctrl\tlog\tckpt" > "${SUMMARY}"

run_window_arm() {
    local arm="$1"
    local desc="$2"
    local qat_enabled="$3"
    local qat_surrogate="$4"
    local log="${RESULTS_DIR}/${arm}_${MODE}_s${SEED}_${TS}.log"

    echo ""
    echo "================================================================"
    echo "  ${arm} (${MODE}) : ${desc}"
    echo "  QAT_ENABLED=${qat_enabled} QAT_SURROGATE=${qat_surrogate}"
    echo "  seed=${SEED} GPUs=${NPROC} iterations=${ITERATIONS}"
    echo "  log: ${log}"
    echo "================================================================"

    env \
        "${BASE_ENV[@]}" \
        QAT_ENABLED="${qat_enabled}" \
        QAT_SURROGATE="${qat_surrogate}" \
        INT6_CATS="mlp,attn,aux" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"

    local params raw int6 step_ms bytes delta ckpt
    params="$(extract_metric 'model_params:\K[0-9]+' "${log}")"
    raw="$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")"
    int6="$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")"
    step_ms="$(extract_metric 'step_avg:\K[0-9.]+' "${log}")"
    bytes="$(extract_metric 'Total submission size int6\+(?:zstd|zlib|brotli): \K[0-9]+' "${log}")"

    [[ -n "${params}" ]] || params="?"
    [[ -n "${raw}" ]] || raw="?"
    [[ -n "${int6}" ]] || int6="?"
    [[ -n "${step_ms}" ]] || step_ms="?"
    [[ -n "${bytes}" ]] || bytes="?"

    if [[ -z "${CONTROL_INT6_WINDOW}" ]]; then
        CONTROL_INT6_WINDOW="${int6}"
    fi
    delta="$(calc_delta "${CONTROL_INT6_WINDOW}" "${int6}")"

    ckpt="-"
    if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
        ckpt="${RESULTS_DIR}/${arm}_${MODE}_s${SEED}_${TS}.final_model.pt"
        cp -f "${REPO_ROOT}/final_model.pt" "${ckpt}"
    fi

    if [[ "${ckpt}" != "-" ]] && is_better "${int6}" "${BEST_WINDOW_INT6}"; then
        BEST_WINDOW_INT6="${int6}"
        BEST_WINDOW_ARM="${arm}"
        BEST_WINDOW_CKPT="${ckpt}"
    fi

    echo -e "WINDOW\t${arm}\t${desc}\t${qat_enabled}\t${qat_surrogate}\tmlp,attn,aux\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${delta}\t${log}\t${ckpt}" >> "${SUMMARY}"
}

run_quant_arm() {
    local arm="$1"
    local desc="$2"
    local int6_cats="$3"
    local log="${RESULTS_DIR}/${arm}_${MODE}_s${SEED}_${TS}.log"

    echo ""
    echo "================================================================"
    echo "  ${arm} (${MODE}) : ${desc}"
    echo "  INT6_CATS=${int6_cats}"
    echo "  seed=${SEED} GPUs=${NPROC} (post-window: SKIP_TRAIN=1)"
    echo "  ckpt: ${BEST_WINDOW_CKPT}"
    echo "  log: ${log}"
    echo "================================================================"

    env \
        "${BASE_ENV[@]}" \
        SKIP_TRAIN=1 \
        INIT_MODEL_PATH="${BEST_WINDOW_CKPT}" \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        QAT_ENABLED=0 \
        QAT_SURROGATE=legacy \
        INT6_CATS="${int6_cats}" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"

    local params raw int6 step_ms bytes delta
    params="$(extract_metric 'model_params:\K[0-9]+' "${log}")"
    raw="$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")"
    int6="$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")"
    step_ms="$(extract_metric 'step_avg:\K[0-9.]+' "${log}")"
    bytes="$(extract_metric 'Total submission size int6\+(?:zstd|zlib|brotli): \K[0-9]+' "${log}")"

    [[ -n "${params}" ]] || params="?"
    [[ -n "${raw}" ]] || raw="?"
    [[ -n "${int6}" ]] || int6="?"
    [[ -n "${step_ms}" ]] || step_ms="?"
    [[ -n "${bytes}" ]] || bytes="?"

    if [[ -z "${CONTROL_INT6_QUANT}" ]]; then
        CONTROL_INT6_QUANT="${int6}"
    fi
    delta="$(calc_delta "${CONTROL_INT6_QUANT}" "${int6}")"

    echo -e "QUANT\t${arm}\t${desc}\t0\tlegacy\t${int6_cats}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${delta}\t${log}\t${BEST_WINDOW_CKPT}" >> "${SUMMARY}"
}

# WINDOW matrix
run_window_arm "BW23W-00" "control: QAT off (BWX-style)" "0" "legacy"
run_window_arm "BW23W-01" "QAT on with legacy surrogate" "1" "legacy"
run_window_arm "BW23W-02" "QAT on with softclamp surrogate" "1" "softclamp"
run_window_arm "BW23W-03" "QAT on with sigmoidste surrogate" "1" "sigmoidste"

if [[ -z "${BEST_WINDOW_CKPT}" || ! -f "${BEST_WINDOW_CKPT}" ]]; then
    echo "ERROR: best window checkpoint missing: ${BEST_WINDOW_CKPT:-(unset)}" >&2
    exit 1
fi

# QUANT matrix on best WINDOW ckpt
run_quant_arm "BW23Q-00" "quant control (existing policy)" "mlp,attn,aux"
run_quant_arm "BW23Q-01" "sensitivity policy: mlp+attn only" "mlp,attn"
run_quant_arm "BW23Q-02" "aggressive policy: attn-only int6" "attn"
run_quant_arm "BW23Q-03" "aggressive policy: mlp-only int6" "mlp"
run_quant_arm "BW23Q-04" "stress policy: all int6 categories" "all"

cat <<TXT

================================================================
BW23 ${MODE} matrix complete.
summary: ${SUMMARY}

Best window arm:
  arm=${BEST_WINDOW_ARM}
  int6_sw_bpb=${BEST_WINDOW_INT6}
  ckpt=${BEST_WINDOW_CKPT}
================================================================
TXT

column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
