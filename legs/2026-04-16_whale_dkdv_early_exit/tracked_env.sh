#!/usr/bin/env bash
# Tracked env for legs/2026-04-16_whale_dkdv_early_exit.
# This is a kernel-only leg. Inherits training config verbatim from
# legs/2026-04-16_whale_fast_kernels/tracked_env.sh so any speedup is
# attributable only to the dkdv mask-split patch in vault_patch.md.
set -euo pipefail

export RUN_ID="${RUN_ID:-whale_dkdv_early_exit_s${SEED:-444}}"

# --- Inherited training config (frozen, do not mutate in this leg) ---
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

# --- Bench shapes for run.sh ---
# Primary shape per task brief; matches Lever A bench plan.
export BENCH_PRIMARY_BHKD="2,8192,8,4,64"
# Secondary shapes spanning the (T, D, GQA) axes the dkdv kernel cares about.
export BENCH_SHAPES="2,8192,8,4,64;2,4096,8,4,64;2,8192,8,8,64;2,8192,8,4,128"

# --- Triton autotune cache control ---
# Flush the cache once before the first benched run after the patch lands so
# stale configs picked for the pre-patch IR cannot bias the comparison.
export WHALE_FLUSH_TRITON_CACHE="${WHALE_FLUSH_TRITON_CACHE:-1}"
