# whale_bwd_persistent_atomic — environment for H5 benchmark.
# Source before running bench_stable.py.

# Pod stack
export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# Target variant: persistent fused dkdv+dq via fp32 atomic_add on dQ.
export WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT:-fused_bwd}

# maxnreg for the fused kernel. Default empty (no maxnreg) because
# maxnreg=224 breaks atomic_add correctness on this Triton 3.6 stack
# (observed ~19005x amplification with the row*1+col*1000 probe).
# Explicitly set WHALE_BWD_FUSED_MAXNREG=<int> only for sweeps.
export WHALE_BWD_FUSED_MAXNREG=${WHALE_BWD_FUSED_MAXNREG:-}

# Bench defaults. Override per-invocation.
export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-15}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-300}
