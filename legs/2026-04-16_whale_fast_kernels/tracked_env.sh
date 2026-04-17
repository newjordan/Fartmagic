#!/usr/bin/env bash
# Edit this file, not the shell command line, when changing the whale_fast_kernels lane.
set -euo pipefail

# Kernel-work leg. The training config is inherited verbatim from
# legs/2026-04-16_whale_pr1493_triton_kernel so that any speedup is attributable
# only to the kernel change.
export RUN_ID="${RUN_ID:-whale_fast_kernels_s${SEED:-444}}"

export NUM_LAYERS=11
export XSA_LAST_N=11
export VOCAB_SIZE=8192
export ROPE_DIMS=16
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35
export PARALLEL_RESIDUAL_START=7
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
