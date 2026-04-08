#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# BW23 4x Quick Sanity
#
# Purpose:
#   Economical, meaningful 4x-GPU test of the new BW23 QAT concept path.
#   Runs 3 arms on the same baseline:
#     - control (QAT off)
#     - QAT legacy surrogate
#     - QAT softclamp surrogate
#
# Usage (from repo root or anywhere):
#   SEED=444 NPROC_PER_NODE=4 PYTHON_BIN="$(which python)" \
#   bash /home/frosty40/sota_crawler/scripts/run_bw23_4x_quick.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LEG_DIR="${REPO_ROOT}/crawler/2026-04-08_BW23_EcoConcept_9F"
TRAIN_PY="${LEG_DIR}/train_gpt.py"
RESULTS_DIR="${LEG_DIR}/results"
mkdir -p "${RESULTS_DIR}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ITERATIONS="${ITERATIONS:-8000}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-300}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-393216}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
EVAL_STRIDE="${EVAL_STRIDE:-2048}"

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
    echo "ERROR: no training shards under ${DATA_PATH}" >&2
    exit 1
fi
if ! compgen -G "${DATA_PATH}/fineweb_val_*.bin" >/dev/null; then
    echo "ERROR: no validation shards under ${DATA_PATH}" >&2
    exit 1
fi

TORCHRUN=("${PYTHON_BIN}" -m torch.distributed.run)

CUDA_INFO="$("${PYTHON_BIN}" - <<'PY'
import sys
import torch
print(f"exe={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={int(torch.cuda.is_available())}")
print(f"cuda_devices={torch.cuda.device_count()}")
PY
)"
echo "${CUDA_INFO}"
echo "tokenizer_path=${TOKENIZER_PATH}"
echo "data_path=${DATA_PATH}"

if ! grep -q "cuda_available=1" <<<"${CUDA_INFO}"; then
    echo "ERROR: CUDA unavailable in ${PYTHON_BIN} env" >&2
    exit 1
fi

CUDA_DEVICES="$(sed -n 's/^cuda_devices=//p' <<<"${CUDA_INFO}")"
if [[ -n "${CUDA_DEVICES}" && "${CUDA_DEVICES}" =~ ^[0-9]+$ ]] && (( NPROC > CUDA_DEVICES )); then
    echo "ERROR: NPROC_PER_NODE=${NPROC} > cuda_devices=${CUDA_DEVICES}" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_DIR}/summary_quick4x_s${SEED}_${TS}.tsv"
echo -e "arm\tdesc\tqat_enabled\tqat_surrogate\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_ctrl\tlog" > "${SUMMARY}"

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
c = float(sys.argv[1]); v = float(sys.argv[2]); d = v - c
print(f"{'+' if d >= 0 else ''}{d:.6f}")
PY
}

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

CONTROL_INT6=""

run_arm() {
    local arm="$1"
    local desc="$2"
    local qat_enabled="$3"
    local qat_surrogate="$4"
    local log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"

    echo ""
    echo "================================================================"
    echo "  ${arm}: ${desc}"
    echo "  QAT_ENABLED=${qat_enabled} QAT_SURROGATE=${qat_surrogate}"
    echo "  seed=${SEED} GPUs=${NPROC} iterations=${ITERATIONS}"
    echo "  log: ${log}"
    echo "================================================================"

    env \
        TOKENIZER_PATH="${TOKENIZER_PATH}" \
        DATA_PATH="${DATA_PATH}" \
        SEED="${SEED}" \
        ITERATIONS="${ITERATIONS}" \
        MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
        WARMDOWN_ITERS="${ITERATIONS}" \
        TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}" \
        VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
        EVAL_STRIDE="${EVAL_STRIDE}" \
        COMPLEMENT_ALPHA=0 \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=2048 \
        ROPE_DIMS=16 \
        SWA_EVERY=50 \
        MTP_NUM_HEADS=0 \
        LATE_QAT_THRESHOLD=0 \
        MATRIX_LR=0.03 \
        TORCHDYNAMO_OPTIMIZE_DDP=0 \
        COMPILE_FULLGRAPH=1 \
        NGRAM_EVAL_ORDER=0 \
        MODEL_DIM=512 \
        USE_CRAWLER=1 \
        NUM_FLAT_LAYERS=9 \
        NUM_CRAWLER_LAYERS=1 \
        CRAWLER_LOOPS=3 \
        CRAWLER_MLP_MULT=6.0 \
        INST_DIM=32 \
        CRAWLER_QUANT_INT8=0 \
        DELTA_NET_HEADS=0 \
        SKIP_EMA=1 \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        MLP_LEAKY_SLOPE=0.5 \
        CRAWLER_MLP_LEAKY_SLOPE=0.5 \
        CRAWLER_MLP_CHOKE_DIM=0 \
        CRAWLER_MLP_CHOKE_SHAPE=flat \
        CRAWLER_MLP_CHOKE_GROUPS=8 \
        CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
        CRAWLER_LOOP_SMEAR=0 \
        CRAWLER_TAP_DIM=0 \
        CRAWLER_TAP_LOOP_SPECIFIC=1 \
        CRAWLER_TAP_LAYERS=all \
        ANCHOR_DIM=0 \
        FLAT_WEIGHT_SHARE=0 \
        QAT_ENABLED="${qat_enabled}" \
        QAT_SURROGATE="${qat_surrogate}" \
        INT6_CATS="mlp,attn,aux" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"

    local raw int6 step_ms bytes delta
    raw="$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")"
    int6="$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")"
    step_ms="$(extract_metric 'step_avg:\K[0-9.]+' "${log}")"
    bytes="$(extract_metric 'Total submission size int6\+(?:zstd|zlib|brotli): \K[0-9]+' "${log}")"
    [[ -n "${raw}" ]] || raw="?"
    [[ -n "${int6}" ]] || int6="?"
    [[ -n "${step_ms}" ]] || step_ms="?"
    [[ -n "${bytes}" ]] || bytes="?"

    if [[ -z "${CONTROL_INT6}" ]]; then
        CONTROL_INT6="${int6}"
    fi
    delta="$(calc_delta "${CONTROL_INT6}" "${int6}")"

    echo -e "${arm}\t${desc}\t${qat_enabled}\t${qat_surrogate}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${delta}\t${log}" >> "${SUMMARY}"
}

run_arm "Q4X-00" "control (QAT off)" "0" "legacy"
run_arm "Q4X-01" "QAT legacy surrogate" "1" "legacy"
run_arm "Q4X-02" "QAT softclamp surrogate" "1" "softclamp"

echo ""
echo "Quick 4x run complete."
echo "summary: ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"

