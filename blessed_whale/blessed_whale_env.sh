#!/usr/bin/env bash
set -euo pipefail

BLESSED_WHALE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BLESSED_WHALE_ANCHOR="${BLESSED_WHALE_DIR}/blessed_whale_anchor.json"
BLESSED_WHALE_FALLBACK_ANCHOR="${BLESSED_WHALE_DIR}/blessed_whale_fallback_anchor.json"
BLESSED_WHALE_HYBRID_B1952_ANCHOR="${BLESSED_WHALE_DIR}/blessed_whale_hybrid_b1952_anchor.json"
BLESSED_WHALE_MATRIX="${BLESSED_WHALE_DIR}/blessed_whale_benchmark_matrix.json"

require_file() {
  [[ -f "$1" ]] || {
    echo "FATAL: required file missing: $1" >&2
    exit 1
  }
}

require_file "${BLESSED_WHALE_ANCHOR}"
require_file "${BLESSED_WHALE_FALLBACK_ANCHOR}"
require_file "${BLESSED_WHALE_HYBRID_B1952_ANCHOR}"
require_file "${BLESSED_WHALE_MATRIX}"

anchor_json() {
  local file="$1"
  local expr="$2"
  jq -r "${expr}" "${file}"
}

matrix_field() {
  local expr="$1"
  jq -r --arg id "${BLESSED_WHALE_BENCHMARK_ID}" ".[\$id]${expr}" "${BLESSED_WHALE_MATRIX}"
}

if [[ -z "${BLESSED_WHALE_BENCHMARK_ID:-}" ]]; then
  echo "FATAL: BLESSED_WHALE_BENCHMARK_ID must be set explicitly" >&2
  exit 1
fi
export BLESSED_WHALE_BENCHMARK_ID
if [[ "$(jq -r --arg id "${BLESSED_WHALE_BENCHMARK_ID}" 'has($id)' "${BLESSED_WHALE_MATRIX}")" != "true" ]]; then
  echo "FATAL: unsupported BLESSED_WHALE_BENCHMARK_ID=${BLESSED_WHALE_BENCHMARK_ID}" >&2
  exit 1
fi

case "$(matrix_field '.kernel_anchor')" in
  blessed)
    SELECTED_ANCHOR="${BLESSED_WHALE_ANCHOR}"
    ;;
  fallback)
    SELECTED_ANCHOR="${BLESSED_WHALE_FALLBACK_ANCHOR}"
    ;;
  hybrid_b1952)
    SELECTED_ANCHOR="${BLESSED_WHALE_HYBRID_B1952_ANCHOR}"
    ;;
  *)
    echo "FATAL: unsupported kernel anchor in blessed whale matrix" >&2
    exit 1
    ;;
esac

export BLESSED_WHALE_DESCRIPTION="$(matrix_field '.description')"
export BLESSED_WHALE_PURPOSE="$(matrix_field '.purpose')"
export BLESSED_WHALE_PROFILE_ID="$(matrix_field '.kernel_profile_id')"
export BLESSED_WHALE_TRAINER_CONTRACT="pr1493_decompressed.py"
export BLESSED_WHALE_ATTN_MODE="$(matrix_field '.attn_mode')"
export BLESSED_WHALE_SELECTED_ANCHOR="${SELECTED_ANCHOR}"

export ITERATIONS="${ITERATIONS:-$(matrix_field '.iterations // 20000')}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-$(matrix_field '.train_log_every // 500')}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-$(matrix_field '.val_loss_every // 4000')}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-$(matrix_field '.max_wallclock_seconds // 600')}"
export GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-$(matrix_field '.gptq_reserve_seconds // 12')}"
export PROFILE_ENABLED="${PROFILE_ENABLED:-$(matrix_field '.profile_enabled // 0')}"
export PROFILE_WAIT_STEPS="${PROFILE_WAIT_STEPS:-$(matrix_field '.profile_wait_steps // 0')}"
export PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-$(matrix_field '.profile_warmup_steps // 0')}"
export PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-$(matrix_field '.profile_active_steps // 0')}"
export PROFILE_REPEAT="${PROFILE_REPEAT:-$(matrix_field '.profile_repeat // 1')}"
export PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-$(matrix_field '.profile_record_shapes // 1')}"
export PROFILE_MEMORY="${PROFILE_MEMORY:-$(matrix_field '.profile_memory // 1')}"
export PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-$(matrix_field '.profile_with_stack // 0')}"
export PROFILE_STOP_AFTER_TRACE="${PROFILE_STOP_AFTER_TRACE:-$(matrix_field '.profile_stop_after_trace // 1')}"
export SKIP_POST_TRAIN="${SKIP_POST_TRAIN:-$(matrix_field '.skip_post_train // 0')}"
export PROFILE_DIR="${PROFILE_DIR:-}"
export DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-$(matrix_field '.disable_torch_compile // 0')}"
export BLESSED_WHALE_HYBRID_WARMSTART_CALLS="${BLESSED_WHALE_HYBRID_WARMSTART_CALLS:-$(matrix_field '.hybrid_warmstart_calls // 0')}"
export LOOP_ONSET_CHECKPOINT_PATH="${LOOP_ONSET_CHECKPOINT_PATH:-}"
export LOOP_ONSET_STOP_AFTER_SAVE="${LOOP_ONSET_STOP_AFTER_SAVE:-$(matrix_field '.loop_onset_stop_after_save // 0')}"
export RESUME_CHECKPOINT_PATH="${RESUME_CHECKPOINT_PATH:-}"

export RUN_ID="${RUN_ID:-blessed_whale_${BLESSED_WHALE_BENCHMARK_ID,,}_s${SEED:-444}}"

# Best donor recipe.
export NUM_LAYERS=12
export XSA_LAST_N=11
export MATRIX_BITS=5
export MODEL_DIM=576
export EMBEDDING_DIM=576
export NUM_HEADS=8
export NUM_KV_HEADS=4
export VOCAB_SIZE=8192
export ROPE_DIMS=16
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35
export PARALLEL_RESIDUAL_START=8
export QK_GAIN_INIT=5.25
export MLP_MULT=4.0
export MATRIX_LR=0.022
export MUON_MOMENTUM=0.99
export MUON_WD=0.095
export EMA_DECAY=0.9965
export WARMDOWN_FRAC=0.72
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_MOMENTUM=0.9
export DATA_DIR=./data

# Shape provenance: the blessed kernel evidence came from a fixed bench shape
# that differs from the current donor if MODEL_DIM / NUM_HEADS changes.
DONOR_HEAD_DIM=$((MODEL_DIM / NUM_HEADS))
export BLESSED_WHALE_BLESSED_HEAD_DIM="${BLESSED_WHALE_BLESSED_HEAD_DIM:-${DONOR_HEAD_DIM}}"
export BLESSED_WHALE_BLESSED_BENCH_SHAPE="${BLESSED_WHALE_BLESSED_BENCH_SHAPE:-2,2048,${NUM_HEADS},${NUM_KV_HEADS},${BLESSED_WHALE_BLESSED_HEAD_DIM}}"

# Selected kernel profile.
export BLESSED_WHALE_STATUS="$(anchor_json "${SELECTED_ANCHOR}" '.status')"
export BLESSED_WHALE_FAMILY_ID="$(anchor_json "${SELECTED_ANCHOR}" '.family_id')"
export BLESSED_WHALE_DQ_TILE="$(anchor_json "${SELECTED_ANCHOR}" '.dq_tile')"
export BLESSED_WHALE_DKDV_TILE="$(anchor_json "${SELECTED_ANCHOR}" '.dkdv_tile')"
export BLESSED_WHALE_DKDV_MAXNREG="$(anchor_json "${SELECTED_ANCHOR}" '.dkdv_maxnreg')"
export BLESSED_WHALE_QMR="$(anchor_json "${SELECTED_ANCHOR}" '.qmr')"
export BLESSED_WHALE_SOURCE_EVIDENCE="$(anchor_json "${SELECTED_ANCHOR}" '.source_evidence')"

# Compatibility metadata for existing kernel tooling.
export WH_KERN_BENCHMARK_ID="${BLESSED_WHALE_BENCHMARK_ID}"
export WH_KERN_KERNEL_PROFILE_ID="${BLESSED_WHALE_PROFILE_ID}"
export WH_KERN_CONTRACT_TRAINER="${BLESSED_WHALE_TRAINER_CONTRACT}"
export WH_KERN_ATTN_MODE="${BLESSED_WHALE_ATTN_MODE}"
export WH_KERN_WINNER_STATUS="${BLESSED_WHALE_STATUS}"
export WH_KERN_WINNER_FAMILY_ID="${BLESSED_WHALE_FAMILY_ID}"
export WH_KERN_WINNER_DQ_TILE="${BLESSED_WHALE_DQ_TILE}"
export WH_KERN_WINNER_DKDV_TILE="${BLESSED_WHALE_DKDV_TILE}"
export WH_KERN_WINNER_DKDV_MAXNREG="${BLESSED_WHALE_DKDV_MAXNREG}"
export WH_KERN_WINNER_QMR="${BLESSED_WHALE_QMR}"
export WH_KERN_WINNER_SOURCE_EVIDENCE="${BLESSED_WHALE_SOURCE_EVIDENCE}"

export WHALE_BWD_VARIANT=fused_delta_tma
export WHALE_FUSED_DELTA_T_MAX=3072
export WHALE_BWD_Q_CONFIG="${BLESSED_WHALE_DQ_TILE}"
export WHALE_BWD_Q_MAXNREG="${BLESSED_WHALE_QMR}"
export WHALE_BWD_KV_TMA_CONFIG="${BLESSED_WHALE_DKDV_TILE}"
export WHALE_BWD_KV_TMA_MAXNREG="${BLESSED_WHALE_DKDV_MAXNREG}"
