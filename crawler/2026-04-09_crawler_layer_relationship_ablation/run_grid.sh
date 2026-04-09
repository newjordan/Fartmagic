#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# crawler_layer_relationship_ablation
#
# 5x4 grid: NUM_FLAT_LAYERS (5-9) x NUM_CRAWLER_LAYERS (1-4)
# 20 arms. Raw BPB at stop step. No quant sweeps.
#
# Usage:
#   SEED=444 NPROC_PER_NODE=2 bash crawler/2026-04-09_crawler_layer_relationship_ablation/run_grid.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-2}"
ITERATIONS="${ITERATIONS:-1000}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3600}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-196608}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
EVAL_STRIDE="${EVAL_STRIDE:-64}"

TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ----------------------------------------------------------------
# Tokenizer/data path auto-detection
# ----------------------------------------------------------------
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
    echo "ERROR: no training shards found under ${DATA_PATH}" >&2
    exit 1
fi

echo "tokenizer_path=${TOKENIZER_PATH}"
echo "data_path=${DATA_PATH}"

# ----------------------------------------------------------------
# Environment probe
# ----------------------------------------------------------------
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
    echo "ERROR: CUDA is not available in ${PYTHON_BIN} environment." >&2
    exit 1
fi

CUDA_DEVICES="$(sed -n 's/^cuda_devices=//p' <<<"${CUDA_INFO}")"
if [[ -n "${CUDA_DEVICES}" && "${CUDA_DEVICES}" =~ ^[0-9]+$ ]] && (( NPROC > CUDA_DEVICES )); then
    echo "WARN: NPROC_PER_NODE=${NPROC} > cuda_devices=${CUDA_DEVICES}; clamping."
    NPROC="${CUDA_DEVICES}"
fi

# ----------------------------------------------------------------
# Results setup
# ----------------------------------------------------------------
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_DIR}/summary_grid_s${SEED}_${TS}.tsv"

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

# ----------------------------------------------------------------
# BWX 9F base env — everything except the two grid knobs
# ----------------------------------------------------------------
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

# TSV header
echo -e "flat\tcrawler\tarm\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tlog" > "${SUMMARY}"

CONTROL_RAW=""

echo ""
echo "========================================"
echo "  crawler_layer_relationship_ablation"
echo "  5x4 grid: flat(5-9) x crawler(1-4)"
echo "  2xGPU, ${ITERATIONS} steps, seed=${SEED}"
echo "========================================"

for FLAT in 9 8 7 6 5; do
    for CRAWL in 1 2 3 4; do
        ARM="${FLAT}F_${CRAWL}C"
        LOG="${RESULTS_DIR}/${ARM}_grid_s${SEED}_${TS}.log"

        echo ""
        echo "================================================================"
        echo "  ${ARM} : NUM_FLAT_LAYERS=${FLAT} NUM_CRAWLER_LAYERS=${CRAWL}"
        echo "  seed=${SEED} GPUs=${NPROC} iterations=${ITERATIONS}"
        echo "  log: ${LOG}"
        echo "================================================================"

        env \
            "${BASE_ENV[@]}" \
            NUM_FLAT_LAYERS="${FLAT}" \
            NUM_CRAWLER_LAYERS="${CRAWL}" \
            "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
            2>&1 | tee "${LOG}"

        PARAMS="$(extract_metric 'model_params:\K[0-9]+' "${LOG}")"
        RAW="$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}")"
        INT6="$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${LOG}")"
        STEP_MS="$(extract_metric 'step_avg:\K[0-9.]+' "${LOG}")"
        BYTES="$(extract_metric 'Total submission size int6\+(?:zstd|zlib|brotli): \K[0-9]+' "${LOG}")"

        [[ -n "${PARAMS}" ]] || PARAMS="?"
        [[ -n "${RAW}" ]] || RAW="?"
        [[ -n "${INT6}" ]] || INT6="?"
        [[ -n "${STEP_MS}" ]] || STEP_MS="?"
        [[ -n "${BYTES}" ]] || BYTES="?"

        if [[ -z "${CONTROL_RAW}" && "${FLAT}" == "9" && "${CRAWL}" == "1" ]]; then
            CONTROL_RAW="${RAW}"
        fi

        echo -e "${FLAT}\t${CRAWL}\t${ARM}\t${PARAMS}\t${RAW}\t${INT6}\t${STEP_MS}\t${BYTES}\t${LOG}" >> "${SUMMARY}"
    done
done

echo ""
echo "========================================"
echo "  Grid complete. Summary: ${SUMMARY}"
echo "  Control (9F+1C) raw_bpb: ${CONTROL_RAW:-?}"
echo "========================================"
echo ""
column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
