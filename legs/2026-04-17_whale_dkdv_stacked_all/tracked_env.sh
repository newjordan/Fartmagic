#!/usr/bin/env bash
# Tracked env for legs/2026-04-17_whale_dkdv_stacked_all.
# Coordination leg that benches the 4-lever stack on whale dkdv.
# Sequenced phases; each phase is invoked by run.sh with its phase name.
set -euo pipefail

export RUN_ID="${RUN_ID:-whale_dkdv_stacked_all_s${SEED:-444}}"

# --- Bench shapes for run.sh (4 primary shapes per hypothesis.md) ---
# Format: "B,T,H,KV,D" separated by ';'.
export BENCH_SHAPES="2,8192,8,4,64;2,3072,8,4,64;2,2048,8,4,64;2,8192,8,4,128"

# --- Routing: force inline-Δ path for ALL shapes (including T=8192) ---
# Default dispatcher cutoff is T<=3072. We override so the primary shape
# routes into _attn_bwd_dkdv_inline_delta_kernel. See vault L1388-L1391.
export WHALE_BWD_VARIANT="${WHALE_BWD_VARIANT:-auto}"
export WHALE_FUSED_DELTA_T_MAX="${WHALE_FUSED_DELTA_T_MAX:-8192}"

# --- Per-phase knobs (default OFF; run.sh overrides per phase) ---
# Lever B — TMA loads for Q/K/V/DO inside dkdv. Default off so pointer-based
# path runs; set to 1 in phase 3+.
export WHALE_DKDV_TMA_LOADS="${WHALE_DKDV_TMA_LOADS:-0}"
# Lever C — persistent dkdv. Default off. Set to 1 in phase 4.
export WHALE_DKDV_PERSISTENT="${WHALE_DKDV_PERSISTENT:-0}"

# --- Triton autotune cache control ---
# Each phase should flush once before its first benched run to prevent
# stale picks from the previous phase's IR.
export WHALE_FLUSH_TRITON_CACHE="${WHALE_FLUSH_TRITON_CACHE:-1}"

# --- Numerics tolerance (reused from fast_kernels leg) ---
export WHALE_NUMERICS_FWD_ATOL="${WHALE_NUMERICS_FWD_ATOL:-5e-3}"
export WHALE_NUMERICS_GRAD_ATOL="${WHALE_NUMERICS_GRAD_ATOL:-1e-2}"

# --- Bench iteration counts ---
export WHALE_BENCH_WARMUP="${WHALE_BENCH_WARMUP:-10}"
export WHALE_BENCH_TIMED="${WHALE_BENCH_TIMED:-50}"
