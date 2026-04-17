#!/usr/bin/env bash
# Tracked env for legs/2026-04-16_whale_dkdv_gqa_grid.
# Re-runs the EXISTING WHALE_BWD_SPLIT_H=1 lever (already in vault) on the
# current vault head (post inline-Δ + early-exit work in flight) to check
# whether the 2026-04-16_whale_pod_autoresearch finding ("split-H is ~20%
# slower") still holds.
set -euo pipefail

export RUN_ID="${RUN_ID:-whale_dkdv_gqa_grid_s${SEED:-444}}"

# --- Inherited training config (frozen, do not mutate in this leg) ---
# Verbatim from legs/2026-04-16_whale_dkdv_early_exit/tracked_env.sh so any
# delta is attributable only to the WHALE_BWD_SPLIT_H toggle.
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

# --- Lever toggle (the variable this leg tests) ---
# Set by run.sh per phase:
#   before -> WHALE_BWD_SPLIT_H=0 (default)
#   after  -> WHALE_BWD_SPLIT_H=1

# --- Bench shapes ---
# Same 4 production shapes as legs/2026-04-16_whale_dkdv_early_exit, plus
# 2 launch-bound synthetic shapes that probe the regime where split-H
# could plausibly help (small program count vs 132 SMs).
export BENCH_PRIMARY_BHKD="2,8192,8,4,64"
export BENCH_SHAPES="2,8192,8,4,64;2,4096,8,4,64;2,8192,8,8,64;2,8192,8,4,128;1,1024,8,2,64;1,2048,8,2,64"

# --- Variant control for split-H to actually activate ---
# WHALE_BWD_VARIANT=auto picks fused-delta when T <= 3072. The split-H
# branch only runs in the baseline (non-fused-delta) path. To exercise
# split-H on the launch-bound short-T shapes (T<=2048) we need to force
# baseline. The 4 long-T production shapes already run baseline under
# `auto`.
export WHALE_BWD_VARIANT="${WHALE_BWD_VARIANT:-baseline}"

# --- Triton autotune cache control ---
# Flush once before the AFTER pass so the new split-H kernel re-tunes
# fresh and doesn't reuse stale entries from prior sessions.
export WHALE_FLUSH_TRITON_CACHE="${WHALE_FLUSH_TRITON_CACHE:-1}"
