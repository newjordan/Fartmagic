# Ablation leg environment. Controls which variant the Triton backward kernels use.
# Source before running bench_stable.py.

# Pod stack
export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# Default variant (iterated during ablation):
#   baseline      — original 3-kernel backward
#   fused_delta   — Δ computed inline inside dq (no preprocess kernel)
#   tma           — fused_delta + TMA descriptors on main loads
#   persistent    — single persistent kernel for all of bwd
export WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT:-baseline}

# Triton autotune override (optional)
export WHALE_BWD_MAXNREG=${WHALE_BWD_MAXNREG:-0}  # 0 = don't force
