# whale_long_t_profile — env for H6.
# Source before run.sh.

export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# WHALE_BWD_VARIANT is set per-phase by run.sh; default here is auto for
# any standalone bench invocations.
export WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT:-auto}

export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-15}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-300}
