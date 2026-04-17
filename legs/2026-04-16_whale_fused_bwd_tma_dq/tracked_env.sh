# whale_fused_bwd_tma_dq — env for H8 (Lever B).
# Source before run.sh.

export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# NEW variant value: routes to the TMA-dq fused bwd kernel proposed in
# vault_patch.md. The dispatch site at vault/whale_kernel_triton.py:1227
# (post-Lever-A) must be extended to recognize this string.
export WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT:-fused_bwd_tma_dq}

# Leave register cap unset (no maxnreg in autotune configs). Scalar
# atomic_add corrupts at maxnreg>=224 on this stack (Triton 3.6 cu130);
# TMA atomic_add lowers via cp.reduce.async.bulk.add and *should* be
# safe, but we don't force it until verified. The patched config list
# pins maxnreg<=192 internally for safety.
export WHALE_BWD_FUSED_MAXNREG=${WHALE_BWD_FUSED_MAXNREG:-}

export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-15}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-300}
