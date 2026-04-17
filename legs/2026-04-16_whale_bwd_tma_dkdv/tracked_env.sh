# whale_bwd_tma_dkdv — environment for H2 benchmark.
# Source before running bench_stable.py.

export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# Target variant: fused_delta with TMA descriptors on dkdv loads.
# (dq stays on the existing _attn_bwd_dq_inline_delta_kernel — TMA only
# applies to dkdv in this leg.)
export WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT:-fused_delta_tma}

export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-15}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-300}
