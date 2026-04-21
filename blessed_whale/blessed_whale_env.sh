#!/usr/bin/env bash
set -euo pipefail

BLESSED_WHALE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BLESSED_WHALE_ANCHOR="${BLESSED_WHALE_DIR}/blessed_whale_anchor.json"
BLESSED_WHALE_FALLBACK_ANCHOR="${BLESSED_WHALE_DIR}/blessed_whale_fallback_anchor.json"
BLESSED_WHALE_MATRIX="${BLESSED_WHALE_DIR}/blessed_whale_benchmark_matrix.json"

require_file() {
  [[ -f "$1" ]] || {
    echo "FATAL: required file missing: $1" >&2
    exit 1
  }
}

require_file "${BLESSED_WHALE_ANCHOR}"
require_file "${BLESSED_WHALE_FALLBACK_ANCHOR}"
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

export BLESSED_WHALE_BENCHMARK_ID="${BLESSED_WHALE_BENCHMARK_ID:-B1}"
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

export RUN_ID="${RUN_ID:-blessed_whale_${BLESSED_WHALE_BENCHMARK_ID,,}_s${SEED:-444}}"

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
