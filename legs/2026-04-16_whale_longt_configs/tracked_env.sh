# whale_longt_configs — env for Lever D (H10).
# Source before run.sh.

export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# Default whale variants for the bench.
export WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT:-auto}
export WHALE_FWD_VARIANT=${WHALE_FWD_VARIANT:-default}

# Bench discipline.
export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-15}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-300}

# New knob — opt into the long-T-biased autotune config lists in
# vault/whale_kernel_triton.py once vault_patch.md is applied. Unset/0 ==
# legacy short-T-biased lists (status quo).
export WHALE_LONGT_CONFIGS=${WHALE_LONGT_CONFIGS:-1}
