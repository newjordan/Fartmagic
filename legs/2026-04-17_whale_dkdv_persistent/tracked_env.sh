# whale_dkdv_persistent — environment for H6 benchmark.
# Source before running bench_stable.py.

export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# H6 target: persistent dkdv kernel.
# Long-T shapes route through baseline (use_fused_delta=False) at
# WHALE_FUSED_DELTA_T_MAX=3072. For the headline 2048 shapes we force
# baseline so the persistent path is exercised on every shape in the
# sweep below.
export WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT:-baseline}
export WHALE_BWD_KV_PERSISTENT=${WHALE_BWD_KV_PERSISTENT:-1}

export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-15}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-300}
