#!/usr/bin/env bash
set -euo pipefail
# ================================================================
# crawler_9f_corpus_ablations_v1
#
# Comprehensive 16-arm screening ablation on BWX 9F base.
# 12 training arms (battery/recurrence + QAT) + 4 quant-only arms.
#
# Usage:
#   SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-08_crawler_9f_corpus_ablations_v1/run_ablation_sequence.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
ITERATIONS="${ITERATIONS:-1500}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3600}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-393216}"
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
if ! compgen -G "${DATA_PATH}/fineweb_val_*.bin" >/dev/null; then
    echo "ERROR: no validation shards found under ${DATA_PATH}" >&2
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
    echo "WARN: NPROC_PER_NODE=${NPROC} > cuda_devices=${CUDA_DEVICES}; auto-clamping."
    NPROC="${CUDA_DEVICES}"
fi

# ----------------------------------------------------------------
# Results setup
# ----------------------------------------------------------------
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_DIR}/summary_screen_s${SEED}_${TS}.tsv"

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
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
print(f"{sign}{d:.8f}")
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

# ----------------------------------------------------------------
# BWX 9F base env (production config)
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
    QAT_ENABLED=0
    QAT_SURROGATE=legacy
    INT6_CATS=mlp,attn,aux
    NPROC_PER_NODE="${NPROC}"
)

CONTROL_INT6=""
CONTROL_QUANT_INT6=""
BEST_CTRL_CKPT=""

echo -e "phase\tarm\tdesc\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_ctrl\tlog\tckpt" > "${SUMMARY}"

# ----------------------------------------------------------------
# Generic training arm runner
# ----------------------------------------------------------------
run_train_arm() {
    local arm="$1"
    local desc="$2"
    shift 2
    # Remaining args are KEY=VALUE overrides
    local -a overrides=("$@")
    local log="${RESULTS_DIR}/${arm}_screen_s${SEED}_${TS}.log"

    echo ""
    echo "================================================================"
    echo "  ${arm} : ${desc}"
    echo "  overrides: ${overrides[*]:-none}"
    echo "  seed=${SEED} GPUs=${NPROC} iterations=${ITERATIONS}"
    echo "  log: ${log}"
    echo "================================================================"

    env \
        "${BASE_ENV[@]}" \
        "${overrides[@]}" \
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

    if [[ -z "${CONTROL_INT6}" ]]; then
        CONTROL_INT6="${int6}"
    fi
    delta="$(calc_delta "${CONTROL_INT6}" "${int6}")"

    ckpt="-"
    if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
        ckpt="${RESULTS_DIR}/${arm}_screen_s${SEED}_${TS}.final_model.pt"
        cp -f "${REPO_ROOT}/final_model.pt" "${ckpt}"
    fi

    # Save control checkpoint for quant phase
    if [[ "${arm}" == "A00" && "${ckpt}" != "-" ]]; then
        BEST_CTRL_CKPT="${ckpt}"
    fi

    echo -e "TRAIN\t${arm}\t${desc}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${delta}\t${log}\t${ckpt}" >> "${SUMMARY}"
}

# ----------------------------------------------------------------
# Quant-only arm runner (post-train, no retrain)
# ----------------------------------------------------------------
run_quant_arm() {
    local arm="$1"
    local desc="$2"
    local int6_cats="$3"
    local log="${RESULTS_DIR}/${arm}_screen_s${SEED}_${TS}.log"

    echo ""
    echo "================================================================"
    echo "  ${arm} : ${desc}"
    echo "  INT6_CATS=${int6_cats}"
    echo "  seed=${SEED} GPUs=${NPROC} (post-train: SKIP_TRAIN=1)"
    echo "  ckpt: ${BEST_CTRL_CKPT}"
    echo "  log: ${log}"
    echo "================================================================"

    env \
        "${BASE_ENV[@]}" \
        SKIP_TRAIN=1 \
        INIT_MODEL_PATH="${BEST_CTRL_CKPT}" \
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

    if [[ -z "${CONTROL_QUANT_INT6}" ]]; then
        CONTROL_QUANT_INT6="${int6}"
    fi
    delta="$(calc_delta "${CONTROL_QUANT_INT6}" "${int6}")"

    echo -e "QUANT\t${arm}\t${desc}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${delta}\t${log}\t${BEST_CTRL_CKPT}" >> "${SUMMARY}"
}

# ================================================================
# TRAINING ARMS (12 arms, sequential)
# ================================================================

echo ""
echo "========================================"
echo "  PHASE 1: Training arms (12 arms)"
echo "  4xGPU, ${ITERATIONS} steps, seed=${SEED}"
echo "========================================"
echo ""

# --- A00: Control (BWX 9F production) ---
run_train_arm "A00" "Control (BWX 9F production)"

# --- Battery / Recurrence ---
run_train_arm "A01" "TAP shared (inter-loop read)" \
    CRAWLER_TAP_DIM=32 \
    CRAWLER_TAP_LOOP_SPECIFIC=0

run_train_arm "A02" "Anchor (inter-loop write)" \
    ANCHOR_DIM=32

run_train_arm "A03" "4 loops naive battery" \
    CRAWLER_LOOPS=4 \
    CRAWLER_LOOP_ROPE_SCALES=9,1,1,1

run_train_arm "A04" "4 loops differentiated battery" \
    CRAWLER_LOOPS=4 \
    CRAWLER_LOOP_ROPE_SCALES=9,3,1,1

run_train_arm "A05" "5 loops progressive battery" \
    CRAWLER_LOOPS=5 \
    CRAWLER_LOOP_ROPE_SCALES=9,5,3,1,1

run_train_arm "A06" "Wider INST_DIM=64" \
    INST_DIM=64

run_train_arm "A07" "2 crawler layers" \
    NUM_CRAWLER_LAYERS=2

run_train_arm "A08" "Crawler int8 quant" \
    CRAWLER_QUANT_INT8=1

# --- QAT Surrogates ---
run_train_arm "A09" "QAT legacy (STE)" \
    QAT_ENABLED=1 \
    QAT_SURROGATE=legacy

run_train_arm "A10" "QAT softclamp" \
    QAT_ENABLED=1 \
    QAT_SURROGATE=softclamp

run_train_arm "A11" "QAT sigmoidste" \
    QAT_ENABLED=1 \
    QAT_SURROGATE=sigmoidste

# ================================================================
# QUANT ARMS (4 arms, post-train on A00 control checkpoint)
# ================================================================

echo ""
echo "========================================"
echo "  PHASE 2: Quant policy arms (4 arms)"
echo "  Post-train on A00 control checkpoint"
echo "========================================"
echo ""

if [[ -z "${BEST_CTRL_CKPT}" || ! -f "${BEST_CTRL_CKPT}" ]]; then
    echo "ERROR: A00 control checkpoint missing: ${BEST_CTRL_CKPT:-(unset)}" >&2
    echo "Skipping quant phase." >&2
else
    run_quant_arm "Q00" "Quant control (existing policy)" "mlp,attn,aux"
    run_quant_arm "Q01" "Drop aux to int8" "mlp,attn"
    run_quant_arm "Q02" "Aggressive: attn-only int6" "attn"
    run_quant_arm "Q03" "Aggressive: mlp-only int6" "mlp"
fi

# ================================================================
# Summary
# ================================================================

cat <<TXT

================================================================
  crawler_9f_corpus_ablations_v1 — screen complete
  seed=${SEED} GPUs=${NPROC} iterations=${ITERATIONS}
  summary: ${SUMMARY}
  control int6_sw_bpb: ${CONTROL_INT6:-?}
================================================================

TXT

column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
