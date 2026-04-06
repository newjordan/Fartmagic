#!/bin/bash
set -euo pipefail
# ================================================================
# BW XII Quant Fix — R4 Full-Fix + Smart Skip Combo
#
# Single-arm follow-up after the QuantFix sweep and 2x GPU movie test.
# This script trains the merged hypothesis:
#   - R4 full fix (fire_embed + inject cap + merge cap)
#   - smart_skip enabled
#   - wd=0.12 (best smart_skip setting from the 1-GPU sweep)
#
# Then it quant-sweeps that exact checkpoint with:
#   Q1: loop-aware GPTQ
#   Q2: loop-aware GPTQ + crawler int8
#   Q3: standard GPTQ
#
# Usage:
#   SEED=444 bash crawler/2026-04-04_BWXII_QuantFix/run_quantfix_r4_smart_combo_1gpu.sh
#
# Useful overrides:
#   AUTO_DATASET=1
#   COMBO_ITERATIONS=2000
#   COMBO_MUON_WD=0.12
#   COMBO_HELIX_INJECT_CAP=1.0
#   COMBO_HELIX_MERGE_CAP=0.7
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/scripts/activate_flywheel_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/scripts/activate_flywheel_env.sh"
fi

if [[ -d "${REPO_ROOT}/flash-attention/hopper" ]]; then
    export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
fi

SEED="${SEED:-444}"
TS="$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${PYTHON_BIN:-python}"
AUTO_DATASET="${AUTO_DATASET:-0}"

COMBO_NAME="${COMBO_NAME:-R4_full_fix_smart}"
COMBO_ITERATIONS="${COMBO_ITERATIONS:-2000}"
COMBO_MAX_WALLCLOCK_SECONDS="${COMBO_MAX_WALLCLOCK_SECONDS:-3600}"
COMBO_GPTQ_CAL_SAMPLES="${COMBO_GPTQ_CAL_SAMPLES:-128}"
COMBO_GPTQ_CAL_SEQ_LEN="${COMBO_GPTQ_CAL_SEQ_LEN:-2048}"

COMBO_MUON_WD="${COMBO_MUON_WD:-0.12}"
COMBO_SMART_SKIP="${COMBO_SMART_SKIP:-1}"
COMBO_HELIX_FIRE_EMBED="${COMBO_HELIX_FIRE_EMBED:-1}"
COMBO_HELIX_INJECT_CAP="${COMBO_HELIX_INJECT_CAP:-1.0}"
COMBO_HELIX_MERGE_CAP="${COMBO_HELIX_MERGE_CAP:-0.7}"

SHARED_DATA_PATH="${SHARED_DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
SHARED_TOKENIZER_PATH="${SHARED_TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
RESULTS_DIR="${SCRIPT_DIR}/results_combo"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/${COMBO_NAME}_summary_s${SEED}_${TS}.tsv"

TRAIN_LOG="${RESULTS_DIR}/${COMBO_NAME}_train_s${SEED}_${TS}.log"
CKPT="${RESULTS_DIR}/${COMBO_NAME}_s${SEED}_${TS}.final_model.pt"
TRAIN_INT6_ARTIFACT="${RESULTS_DIR}/${COMBO_NAME}_train_s${SEED}_${TS}.final_model.int6.ptz"

Q1_LOG="${RESULTS_DIR}/${COMBO_NAME}_q1_s${SEED}_${TS}.log"
Q1_ARTIFACT="${RESULTS_DIR}/${COMBO_NAME}_q1_s${SEED}_${TS}.final_model.int6.ptz"

Q2_LOG="${RESULTS_DIR}/${COMBO_NAME}_q2_s${SEED}_${TS}.log"
Q2_ARTIFACT="${RESULTS_DIR}/${COMBO_NAME}_q2_s${SEED}_${TS}.final_model.int6.ptz"

Q3_LOG="${RESULTS_DIR}/${COMBO_NAME}_q3_s${SEED}_${TS}.log"
Q3_ARTIFACT="${RESULTS_DIR}/${COMBO_NAME}_q3_s${SEED}_${TS}.final_model.int6.ptz"

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" 2>/dev/null | tail -1 || echo "?"
}

calc_gap() {
    local raw="$1"
    local int6="$2"
    "${PYTHON_BIN}" - <<PY
r = "${raw}"
i = "${int6}"
if r != "?" and i != "?":
    print(f"{float(i) - float(r):.4f}")
else:
    print("?")
PY
}

common_exports() {
    export DATA_PATH="${SHARED_DATA_PATH}"
    export TOKENIZER_PATH="${SHARED_TOKENIZER_PATH}"

    export SEED="${SEED}"
    export ITERATIONS="${COMBO_ITERATIONS}"
    export MAX_WALLCLOCK_SECONDS="${COMBO_MAX_WALLCLOCK_SECONDS}"
    export TRAIN_BATCH_TOKENS=786432
    export WARMDOWN_ITERS=2000
    export COMPLEMENT_ALPHA=0
    export XSA_LAST_N=11
    export BIGRAM_VOCAB_SIZE=2048
    export ROPE_DIMS=16
    export SWA_EVERY=50
    export MTP_NUM_HEADS=0
    export LATE_QAT_THRESHOLD=0
    export MATRIX_LR=0.03
    export EMBED_LR=0.035
    export TORCHDYNAMO_OPTIMIZE_DDP=0
    export COMPILE_FULLGRAPH=1
    export NGRAM_EVAL_ORDER=0
    export MODEL_DIM=512
    export USE_CRAWLER=1
    export NUM_FLAT_LAYERS=9
    export NUM_CRAWLER_LAYERS=1
    export CRAWLER_LOOPS=1
    export CRAWLER_MLP_MULT=6.0
    export INST_DIM=32
    export DELTA_NET_HEADS=0
    export SKIP_EMA=1
    export QK_GAIN_INIT=4.0
    export GPTQ_CAL_SAMPLES="${COMBO_GPTQ_CAL_SAMPLES}"
    export GPTQ_CAL_SEQ_LEN="${COMBO_GPTQ_CAL_SEQ_LEN}"
    export MLP_LEAKY_SLOPE=0.5
    export CRAWLER_MLP_LEAKY_SLOPE=0.5
    export CRAWLER_MLP_CHOKE_DIM=0
    export CRAWLER_MLP_CHOKE_SHAPE=flat
    export CRAWLER_MLP_CHOKE_GROUPS=8
    export CRAWLER_LOOP_ROPE_SCALES=9,1,1
    export CRAWLER_LOOP_SMEAR=0
    export CRAWLER_TAP_DIM=0
    export CRAWLER_TAP_LOOP_SPECIFIC=1
    export CRAWLER_TAP_LAYERS=all
    export ANCHOR_DIM=0
    export FLAT_WEIGHT_SHARE=0
    export HELIX=1
    export HELIX_DIM=192
    export HELIX_STRIDE=1
    export CRAWLER_CROSS_HEADS=4
    export MUON_WD="${COMBO_MUON_WD}"
    export SMART_SKIP="${COMBO_SMART_SKIP}"
    export HELIX_FIRE_EMBED="${COMBO_HELIX_FIRE_EMBED}"
    export HELIX_INJECT_CAP="${COMBO_HELIX_INJECT_CAP}"
    export HELIX_MERGE_CAP="${COMBO_HELIX_MERGE_CAP}"
    export NPROC_PER_NODE=1
}

preflight() {
    log "Preflight: environment"
    VERIFY_DATA=0 bash "${REPO_ROOT}/scripts/verify_cu124_fa3_env.sh"

    "${PYTHON_BIN}" -c "import brotli; print('brotli OK')" >/dev/null 2>&1 \
        || { log "Installing brotli"; "${PYTHON_BIN}" -m pip install brotli -q; }

    "${PYTHON_BIN}" - <<'PY'
try:
    import flash_attn_interface
    print("FA3 (hopper) OK")
except Exception:
    try:
        import flash_attn
        print(f"flash-attn v{flash_attn.__version__}")
    except Exception as exc:
        raise SystemExit(f"ERROR: flash-attn not importable: {exc}")
PY

    if [[ ! -f "${SHARED_TOKENIZER_PATH}" ]]; then
        if [[ "${AUTO_DATASET}" == "1" ]]; then
            log "Tokenizer missing; downloading challenge artifacts"
            "${PYTHON_BIN}" data/cached_challenge_fineweb.py --variant sp1024 --train-shards 8
        else
            echo "ERROR: tokenizer missing at ${SHARED_TOKENIZER_PATH}" >&2
            echo "Run: python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 8" >&2
            exit 1
        fi
    fi

    local shard_count
    shard_count="$(
        DATA_PATH="${SHARED_DATA_PATH}" "${PYTHON_BIN}" - <<'PY'
import glob
import os
data_path = os.environ["DATA_PATH"]
print(len(glob.glob(os.path.join(data_path, "fineweb_train_*.bin"))))
PY
    )"

    if (( shard_count < 8 )); then
        if [[ "${AUTO_DATASET}" == "1" ]]; then
            log "Only ${shard_count} train shards found; downloading 8 shards"
            "${PYTHON_BIN}" data/cached_challenge_fineweb.py --variant sp1024 --train-shards 8
        else
            echo "ERROR: need >=8 train shards, found ${shard_count}" >&2
            echo "Run: python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 8" >&2
            exit 1
        fi
    fi

    log "Preflight: dataset OK"
}

run_train() {
    log "Train: ${COMBO_NAME}"
    common_exports
    export RUN_ID="${COMBO_NAME}_train_${TS}"
    export SKIP_GPTQ=1
    export LOOP_AWARE_GPTQ=0
    export CRAWLER_QUANT_INT8=0
    export SKIP_TRAIN=0
    export INIT_MODEL_PATH=""
    "${PYTHON_BIN}" "${TRAIN_PY}" 2>&1 | tee "${TRAIN_LOG}"
    cp -f "${REPO_ROOT}/final_model.pt" "${CKPT}"
    cp -f "${REPO_ROOT}/final_model.int6.ptz" "${TRAIN_INT6_ARTIFACT}"
}

run_quant() {
    local quant_name="$1"
    local loop_aware="$2"
    local crawler_int8="$3"
    local qlog="$4"
    local artifact="$5"

    log "Quant: ${quant_name}"
    common_exports
    export RUN_ID="${COMBO_NAME}_${quant_name}_${TS}"
    export SKIP_GPTQ=0
    export LOOP_AWARE_GPTQ="${loop_aware}"
    export CRAWLER_QUANT_INT8="${crawler_int8}"
    export SKIP_TRAIN=1
    export INIT_MODEL_PATH="${CKPT}"
    "${PYTHON_BIN}" "${TRAIN_PY}" 2>&1 | tee "${qlog}"
    cp -f "${REPO_ROOT}/final_model.int6.ptz" "${artifact}"
}

write_row() {
    local phase="$1"
    local quant_method="$2"
    local logfile="$3"
    local artifact="$4"
    local raw_bpb
    local int6_bpb
    local bytes
    local gap

    if [[ "${phase}" == "train" ]]; then
        raw_bpb="$(extract_metric 'DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}")"
        int6_bpb="$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}")"
    else
        raw_bpb="$(extract_metric 'DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}")"
        int6_bpb="$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}")"
    fi

    bytes="$(extract_metric 'Total submission size int6\+(?:brotli|zlib): \K[0-9]+' "${logfile}")"
    gap="$(calc_gap "${raw_bpb}" "${int6_bpb}")"

    echo -e "${phase}\t${COMBO_NAME}\t${quant_method}\t${raw_bpb}\t${int6_bpb}\t${gap}\t${bytes}\t${logfile}\t${CKPT}\t${artifact}" >> "${SUMMARY}"
}

{
    echo -e "phase\tarm\tquant_method\traw_bpb\tint6_sw_bpb\tquant_gap\tbytes\tlog\tcheckpoint\tartifact"
} > "${SUMMARY}"

preflight
run_train
write_row "train" "naive_int6" "${TRAIN_LOG}" "${TRAIN_INT6_ARTIFACT}"

run_quant "q1" "1" "0" "${Q1_LOG}" "${Q1_ARTIFACT}"
write_row "quant" "loop-aware GPTQ" "${Q1_LOG}" "${Q1_ARTIFACT}"

run_quant "q2" "1" "1" "${Q2_LOG}" "${Q2_ARTIFACT}"
write_row "quant" "loop-aware+int8" "${Q2_LOG}" "${Q2_ARTIFACT}"

run_quant "q3" "0" "0" "${Q3_LOG}" "${Q3_ARTIFACT}"
write_row "quant" "standard GPTQ" "${Q3_LOG}" "${Q3_ARTIFACT}"

echo ""
echo "================================================================"
echo "  BW XII QUANT FIX — R4 FULL-FIX + SMART SKIP COMPLETE"
echo "  seed=${SEED}"
echo "  combo=${COMBO_NAME}"
echo "  summary: ${SUMMARY}"
echo "  results: ${RESULTS_DIR}"
echo "================================================================"
echo ""
column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
