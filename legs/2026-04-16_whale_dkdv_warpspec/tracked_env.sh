# whale_dkdv_warpspec — env for H7 (Lever A: warp_specialize on dkdv M-loop).
# Source before run.sh.

export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# Force baseline path (the kernel we are patching). `auto` routes T>3072
# to baseline anyway, but we pin it so the bench cannot silently flip
# to fused_delta on a smaller shape.
export WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT:-baseline}

# H7 lever knob (NEW): gate the warp_specialize=True M-loop in
# _attn_bwd_dkdv_kernel. Off (0) preserves the existing baseline path
# byte-for-byte; on (1) enables the warp-specialized loop.
export WHALE_BWD_KV_WARPSPEC=${WHALE_BWD_KV_WARPSPEC:-1}

# Bench discipline (matches H6 long_t_profile): 15 rounds × 300 iters.
export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-15}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-300}
