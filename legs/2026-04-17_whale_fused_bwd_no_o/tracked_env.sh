#!/usr/bin/env bash
# whale_fused_bwd_no_o — environment for A/B bench of fused_bwd variants
# with and without in-kernel O reload.
#
# Requires the vault_patch.md edits to be applied first. Until applied,
# WHALE_BWD_VARIANT=fused_bwd_no_o will fall through to the unknown-variant
# branch (existing code path sets use_fused_delta = variant == "fused_delta"),
# which would silently use the baseline 3-kernel path — so DO NOT launch
# this leg until the patch is in vault/.

# Pod stack (matches parent leg 2026-04-16_whale_bwd_persistent_atomic).
export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# Variant under test. The bench driver overrides per pass.
export WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT:-fused_bwd_no_o}

# Let fused paths run at all T (default auto-switches at T>3072).
export WHALE_FUSED_DELTA_T_MAX=${WHALE_FUSED_DELTA_T_MAX:-1048576}

# Keep maxnreg unset — 224 corrupts atomic_add on Triton 3.6.
# (See user MEMORY: triton_gotchas_atomic_add.md and H5 RESULTS.md.)
export WHALE_BWD_FUSED_MAXNREG=${WHALE_BWD_FUSED_MAXNREG:-}

# Bench defaults.
export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-15}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-300}
