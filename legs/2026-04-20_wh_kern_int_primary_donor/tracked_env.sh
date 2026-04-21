#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WINNER_ANCHOR="${LEG_DIR}/../2026-04-18_whale_cross_gpu_validation_prep_t2048/winner_anchor.json"
FALLBACK_ANCHOR="${LEG_DIR}/fallback_anchor.json"
BENCHMARK_MATRIX="${LEG_DIR}/benchmark_matrix.json"

require_file() {
  [[ -f "$1" ]] || {
    echo "FATAL: required file missing: $1" >&2
    exit 1
  }
}

require_file "${WINNER_ANCHOR}"
require_file "${FALLBACK_ANCHOR}"
require_file "${BENCHMARK_MATRIX}"

anchor_json() {
  local file="$1"
  local expr="$2"
  jq -r "${expr}" "${file}"
}

benchmark_field() {
  local expr="$1"
  jq -r --arg id "${WH_KERN_BENCHMARK_ID}" ".[\$id]${expr}" "${BENCHMARK_MATRIX}"
}

export WH_KERN_BENCHMARK_ID="${WH_KERN_BENCHMARK_ID:-B1}"
if [[ "$(jq -r --arg id "${WH_KERN_BENCHMARK_ID}" 'has($id)' "${BENCHMARK_MATRIX}")" != "true" ]]; then
  echo "FATAL: unsupported WH_KERN_BENCHMARK_ID=${WH_KERN_BENCHMARK_ID}" >&2
  exit 1
fi

case "$(benchmark_field '.kernel_anchor')" in
  winner_anchor)
    SELECTED_ANCHOR="${WINNER_ANCHOR}"
    ;;
  fallback_anchor)
    SELECTED_ANCHOR="${FALLBACK_ANCHOR}"
    ;;
  *)
    echo "FATAL: unsupported kernel anchor in benchmark matrix" >&2
    exit 1
    ;;
esac

export WH_KERN_BENCHMARK_DESC="$(benchmark_field '.description')"
export WH_KERN_BENCHMARK_PURPOSE="$(benchmark_field '.purpose')"
export WH_KERN_KERNEL_PROFILE_ID="$(benchmark_field '.kernel_profile_id')"
export WH_KERN_CONTRACT_TRAINER="pr1493_decompressed.py"
export WH_KERN_BENCHMARK_ANCHOR_FILE="${SELECTED_ANCHOR}"
export WH_KERN_BLESSED_ANCHOR_FILE="${WINNER_ANCHOR}"
export WH_KERN_ATTN_MODE="$(benchmark_field '.attn_mode')"

export RUN_ID="${RUN_ID:-wh_kern_int_${WH_KERN_BENCHMARK_ID,,}_s${SEED:-444}}"

# Best donor recipe.
export NUM_LAYERS=12
export XSA_LAST_N=11
export MATRIX_BITS=5
export MODEL_DIM=576
export EMBEDDING_DIM=576
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

# Selected kernel profile for this benchmark.
export WH_KERN_WINNER_STATUS="$(anchor_json "${SELECTED_ANCHOR}" '.status')"
export WH_KERN_WINNER_FAMILY_ID="$(anchor_json "${SELECTED_ANCHOR}" '.family_id')"
export WH_KERN_WINNER_DQ_TILE="$(anchor_json "${SELECTED_ANCHOR}" '.dq_tile')"
export WH_KERN_WINNER_DKDV_TILE="$(anchor_json "${SELECTED_ANCHOR}" '.dkdv_tile')"
export WH_KERN_WINNER_DKDV_MAXNREG="$(anchor_json "${SELECTED_ANCHOR}" '.dkdv_maxnreg')"
export WH_KERN_WINNER_QMR="$(anchor_json "${SELECTED_ANCHOR}" '.qmr')"
export WH_KERN_WINNER_SOURCE_EVIDENCE="$(anchor_json "${SELECTED_ANCHOR}" '.source_evidence')"

export WHALE_BWD_VARIANT=fused_delta_tma
export WHALE_FUSED_DELTA_T_MAX=3072
export WHALE_BWD_Q_CONFIG="${WH_KERN_WINNER_DQ_TILE}"
export WHALE_BWD_Q_MAXNREG="${WH_KERN_WINNER_QMR}"
export WHALE_BWD_KV_TMA_CONFIG="${WH_KERN_WINNER_DKDV_TILE}"
export WHALE_BWD_KV_TMA_MAXNREG="${WH_KERN_WINNER_DKDV_MAXNREG}"
