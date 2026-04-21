#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WINNER_ANCHOR="${LEG_DIR}/../2026-04-18_whale_cross_gpu_validation_prep_t2048/winner_anchor.json"

anchor_json() {
  jq -r "$1" "${WINNER_ANCHOR}"
}

export RUN_ID="${RUN_ID:-wh_kern_int_primary_donor_s${SEED:-444}}"

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

# Blessed 2x whale kernel carry-forward.
export WH_KERN_WINNER_STATUS="$(anchor_json '.status')"
export WH_KERN_WINNER_FAMILY_ID="$(anchor_json '.family_id')"
export WH_KERN_WINNER_DQ_TILE="$(anchor_json '.dq_tile')"
export WH_KERN_WINNER_DKDV_TILE="$(anchor_json '.dkdv_tile')"
export WH_KERN_WINNER_DKDV_MAXNREG="$(anchor_json '.dkdv_maxnreg')"
export WH_KERN_WINNER_QMR="$(anchor_json '.qmr')"
export WH_KERN_WINNER_SOURCE_EVIDENCE="$(anchor_json '.source_evidence')"

export WHALE_BWD_VARIANT=fused_delta_tma
export WHALE_FUSED_DELTA_T_MAX=3072
export WHALE_BWD_Q_CONFIG="${WH_KERN_WINNER_DQ_TILE}"
export WHALE_BWD_Q_MAXNREG="${WH_KERN_WINNER_QMR}"
export WHALE_BWD_KV_TMA_CONFIG="${WH_KERN_WINNER_DKDV_TILE}"
export WHALE_BWD_KV_TMA_MAXNREG="${WH_KERN_WINNER_DKDV_MAXNREG}"
