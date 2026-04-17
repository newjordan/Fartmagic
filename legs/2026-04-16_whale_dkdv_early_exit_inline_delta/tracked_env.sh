#!/usr/bin/env bash
# Tracked env for legs/2026-04-16_whale_dkdv_early_exit_inline_delta.
# This is a kernel-only leg targeting `_attn_bwd_dkdv_inline_delta_kernel`.
# Inherits training config verbatim from
# legs/2026-04-16_whale_fast_kernels/tracked_env.sh so any speedup is
# attributable only to the dkdv-inline-delta mask-split patch in vault_patch.md.
set -euo pipefail

export RUN_ID="${RUN_ID:-whale_dkdv_early_exit_inline_delta_s${SEED:-444}}"

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
# Forced into the inline-delta path via WHALE_FUSED_DELTA_T_MAX=8192 below.
export BENCH_PRIMARY_BHKD="2,8192,8,4,64"
# Secondary shapes that auto-route to the inline-delta path under default
# WHALE_FUSED_DELTA_T_MAX=3072 (we override anyway, but include them so the
# benched coverage matches the production decision boundary).
export BENCH_SHAPES="2,8192,8,4,64;2,3072,8,4,64;2,2048,8,4,64;2,8192,8,4,128"

# --- Inline-delta routing ---
# Force the inline-delta dkdv kernel for ALL benched T values (default cutoff
# is T<=3072). This exercises the patched kernel at the long-T primary
# shape (T=8192) so we see the full mask-split benefit.
# See vault/whale_kernel_triton.py L1255-L1257 for the dispatcher.
export WHALE_BWD_VARIANT="${WHALE_BWD_VARIANT:-auto}"
export WHALE_FUSED_DELTA_T_MAX="${WHALE_FUSED_DELTA_T_MAX:-8192}"

# --- Triton autotune cache control ---
# Flush the cache once before the first benched run after the patch lands so
# stale configs picked for the pre-patch IR cannot bias the comparison.
export WHALE_FLUSH_TRITON_CACHE="${WHALE_FLUSH_TRITON_CACHE:-1}"
