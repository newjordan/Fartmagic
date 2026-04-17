#!/usr/bin/env bash
# Tracked env for legs/2026-04-17_whale_dkdv_tma_loads.
# Kernel-only leg stacked on top of the early-exit kernel (leg
# 2026-04-16_whale_dkdv_early_exit). Shapes mirror Lever A's bench plan.
set -euo pipefail

export RUN_ID="${RUN_ID:-whale_dkdv_tma_loads_s${SEED:-444}}"

# --- Inherited training config (frozen, do not mutate in this leg) ---
# Mirrors legs/2026-04-16_whale_dkdv_early_exit/tracked_env.sh so any delta
# observed here is attributable only to the TMA-loads patch in vault_patch.md.
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

# --- Bench shapes for run.sh (same 4 primary shapes as Lever A) ---
export BENCH_PRIMARY_BHKD="2,8192,8,4,64"
export BENCH_SHAPES="2,8192,8,4,64;2,4096,8,4,64;2,8192,8,8,64;2,8192,8,4,128"

# --- Triton autotune cache control ---
# The TMA kernel is a new kernel key; flush the cache once before the first
# benched run to avoid stale configs picked for the pre-patch IR.
export WHALE_FLUSH_TRITON_CACHE="${WHALE_FLUSH_TRITON_CACHE:-1}"

# --- Opt-in toggle consumed by custom_whale_attn_bwd ---
# Default is 0 so any caller sourcing this file and then running an
# un-patched vault does NOT crash on an unknown env-key. The run.sh script
# flips this to 1 only in the `after` phase.
export WHALE_BWD_KV_TMA_LOADS="${WHALE_BWD_KV_TMA_LOADS:-0}"
