#!/bin/bash
set -euo pipefail
# ================================================================
# BW XII Quant Fix — 2-GPU Movie Test
#
# Fastest useful Round 2 check on a 2x GPU pod:
#   GPU 0: R1_fire_embed   -> train + loop-aware GPTQ
#   GPU 1: R4_full_fix     -> train + loop-aware GPTQ
#
# Why a separate launcher:
# - train_gpt.py writes final_model.pt and final_model.int6.ptz in the
#   repo root, so two concurrent runs in one checkout clobber each other.
# - This script uses two isolated git worktrees and copies results back to
#   crawler/2026-04-04_BWXII_QuantFix/results_movie/.
#
# Usage:
#   SEED=444 bash crawler/2026-04-04_BWXII_QuantFix/run_quantfix_movie_2gpu.sh
#
# Fresh pod prep:
#   bash scripts/pod_setup.sh
#   source scripts/activate_flywheel_env.sh
#   python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 8
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
KEEP_WORKTREES="${KEEP_WORKTREES:-1}"

MOVIE_ITERATIONS="${MOVIE_ITERATIONS:-2000}"
MOVIE_MAX_WALLCLOCK_SECONDS="${MOVIE_MAX_WALLCLOCK_SECONDS:-3600}"
MOVIE_GPTQ_CAL_SAMPLES="${MOVIE_GPTQ_CAL_SAMPLES:-128}"
MOVIE_GPTQ_CAL_SEQ_LEN="${MOVIE_GPTQ_CAL_SEQ_LEN:-2048}"
SHARED_DATA_PATH="${SHARED_DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
SHARED_TOKENIZER_PATH="${SHARED_TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

RESULTS_DIR="${SCRIPT_DIR}/results_movie"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/movie_summary_s${SEED}_${TS}.tsv"

WORKTREE_PARENT="${WORKTREE_PARENT:-$(dirname "${REPO_ROOT}")}"
WT0="${WORKTREE_PARENT}/bwxii_movie_gpu0_${TS}"
WT1="${WORKTREE_PARENT}/bwxii_movie_gpu1_${TS}"

ARM0_NAME="R1_fire_embed"
ARM0_FIRE="1"
ARM0_INJECT="0"
ARM0_MERGE="0"

ARM1_NAME="R4_full_fix"
ARM1_FIRE="1"
ARM1_INJECT="1.0"
ARM1_MERGE="0.7"

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" 2>/dev/null | tail -1 || echo "?"
}

cleanup() {
    if [[ "${KEEP_WORKTREES}" == "1" ]]; then
        return
    fi
    for wt in "${WT0}" "${WT1}"; do
        if [[ -d "${wt}" ]]; then
            git -C "${REPO_ROOT}" worktree remove --force "${wt}" >/dev/null 2>&1 || true
            rm -rf "${wt}" >/dev/null 2>&1 || true
        fi
    done
}
trap cleanup EXIT

preflight() {
    log "Preflight: environment"
    VERIFY_DATA=0 bash "${REPO_ROOT}/scripts/verify_fa3_env.sh"

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

prepare_worktree() {
    local wt="$1"
    git -C "${REPO_ROOT}" worktree add --detach "${wt}" HEAD >/dev/null

    local name
    for name in data wheels flash-attention; do
        if [[ -e "${REPO_ROOT}/${name}" && ! -e "${wt}/${name}" ]]; then
            ln -s "${REPO_ROOT}/${name}" "${wt}/${name}"
        fi
    done
}

run_arm() {
    local gpu="$1"
    local wt="$2"
    local arm="$3"
    local fire="$4"
    local inject="$5"
    local merge="$6"

    local train_log="${RESULTS_DIR}/${arm}_train_s${SEED}_${TS}.log"
    local quant_log="${RESULTS_DIR}/${arm}_q1_s${SEED}_${TS}.log"
    local ckpt="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.final_model.pt"
    local int6_artifact="${RESULTS_DIR}/${arm}_q1_s${SEED}_${TS}.final_model.int6.ptz"
    local train_py="${wt}/crawler/2026-04-04_BWXII_QuantFix/train_gpt.py"

    (
        set -euo pipefail
        cd "${wt}"

        if [[ -d "${wt}/flash-attention/hopper" ]]; then
            export PYTHONPATH="${wt}/flash-attention/hopper:${PYTHONPATH:-}"
        fi

        export CUDA_VISIBLE_DEVICES="${gpu}"
        export DATA_PATH="${SHARED_DATA_PATH}"
        export TOKENIZER_PATH="${SHARED_TOKENIZER_PATH}"

        export SEED="${SEED}"
        export ITERATIONS="${MOVIE_ITERATIONS}"
        export MAX_WALLCLOCK_SECONDS="${MOVIE_MAX_WALLCLOCK_SECONDS}"
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
        export GPTQ_CAL_SAMPLES="${MOVIE_GPTQ_CAL_SAMPLES}"
        export GPTQ_CAL_SEQ_LEN="${MOVIE_GPTQ_CAL_SEQ_LEN}"
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
        export MUON_WD=0.12
        export SMART_SKIP=0
        export NPROC_PER_NODE=1

        export HELIX_FIRE_EMBED="${fire}"
        export HELIX_INJECT_CAP="${inject}"
        export HELIX_MERGE_CAP="${merge}"

        log "GPU${gpu}: ${arm} train start"
        export RUN_ID="${arm}_train_${TS}"
        export SKIP_GPTQ=1
        export LOOP_AWARE_GPTQ=0
        export CRAWLER_QUANT_INT8=0
        export SKIP_TRAIN=0
        export INIT_MODEL_PATH=""
        "${PYTHON_BIN}" "${train_py}" 2>&1 | tee "${train_log}"
        cp -f final_model.pt "${ckpt}"

        log "GPU${gpu}: ${arm} loop-aware GPTQ start"
        export RUN_ID="${arm}_q1_${TS}"
        export SKIP_GPTQ=0
        export LOOP_AWARE_GPTQ=1
        export CRAWLER_QUANT_INT8=0
        export SKIP_TRAIN=1
        export INIT_MODEL_PATH="${ckpt}"
        "${PYTHON_BIN}" "${train_py}" 2>&1 | tee "${quant_log}"
        cp -f final_model.int6.ptz "${int6_artifact}"

        log "GPU${gpu}: ${arm} complete"
    )
}

write_summary() {
    {
        echo -e "arm\traw_bpb\tint6_sw_bpb\tquant_gap\tbytes\ttrain_log\tquant_log\tcheckpoint\tartifact"
        local arm
        for arm in "${ARM0_NAME}" "${ARM1_NAME}"; do
            local train_log="${RESULTS_DIR}/${arm}_train_s${SEED}_${TS}.log"
            local quant_log="${RESULTS_DIR}/${arm}_q1_s${SEED}_${TS}.log"
            local ckpt="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.final_model.pt"
            local int6_artifact="${RESULTS_DIR}/${arm}_q1_s${SEED}_${TS}.final_model.int6.ptz"
            local raw_bpb
            local int6_bpb
            local bytes
            local gap

            raw_bpb="$(extract_metric 'DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${quant_log}")"
            int6_bpb="$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${quant_log}")"
            bytes="$(extract_metric 'Total submission size int6\+(?:brotli|zlib): \K[0-9]+' "${quant_log}")"
            gap="$("${PYTHON_BIN}" - <<PY
r = "${raw_bpb}"
i = "${int6_bpb}"
if r != "?" and i != "?":
    print(f"{float(i) - float(r):.4f}")
else:
    print("?")
PY
)"

            echo -e "${arm}\t${raw_bpb}\t${int6_bpb}\t${gap}\t${bytes}\t${train_log}\t${quant_log}\t${ckpt}\t${int6_artifact}"
        done
    } > "${SUMMARY}"
}

preflight

log "Preparing isolated worktrees"
prepare_worktree "${WT0}"
prepare_worktree "${WT1}"

log "Launching 2-GPU movie test"
run_arm 0 "${WT0}" "${ARM0_NAME}" "${ARM0_FIRE}" "${ARM0_INJECT}" "${ARM0_MERGE}" &
PID0=$!
run_arm 1 "${WT1}" "${ARM1_NAME}" "${ARM1_FIRE}" "${ARM1_INJECT}" "${ARM1_MERGE}" &
PID1=$!

STATUS0=0
STATUS1=0
wait "${PID0}" || STATUS0=$?
wait "${PID1}" || STATUS1=$?

if (( STATUS0 != 0 || STATUS1 != 0 )); then
    echo "ERROR: movie test failed (gpu0=${STATUS0} gpu1=${STATUS1})" >&2
    exit 1
fi

write_summary

echo ""
echo "================================================================"
echo "  BW XII QUANT FIX — 2-GPU MOVIE TEST COMPLETE"
echo "  seed=${SEED}"
echo "  summary: ${SUMMARY}"
echo "  results: ${RESULTS_DIR}"
echo "  worktrees kept: ${KEEP_WORKTREES}"
echo "================================================================"
echo ""
column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
