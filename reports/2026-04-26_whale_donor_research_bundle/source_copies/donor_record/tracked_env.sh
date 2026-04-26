#!/usr/bin/env bash
# Edit this file, not the shell command line, when changing the whale lane.
# This leg is a silo. If the faithful PR1493 contract changes, create a new leg.
set -euo pipefail

export RUN_ID="${RUN_ID:-whale_l12_dim576_matrixbits5_s${SEED:-444}}"

# Whale variant with width bump, 12 layers, and matrix compression:
# - same PR1493 geometry and TTT settings otherwise
# - 12 layers with the established 12L XSA span
# - matrix banks at 5 bits
# - residual path starts at 8 to preserve the whale's last-four-layers shape
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
