# whale_dkdv_autotune_expand — env
# Source before run.sh.
#
# Pins the variable under test: WHALE_BWD_VARIANT=baseline (Phase-1 winner
# at long T; see legs/2026-04-16_whale_long_t_profile/hypothesis.md).
# No WHALE_BWD_KV_CONFIG override — we WANT autotune to run over the
# expanded config list.

export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# Variable under test.
export WHALE_BWD_VARIANT=baseline

# Let autotune pick from the expanded config list.
unset WHALE_BWD_KV_CONFIG
unset WHALE_BWD_KV_MAXNREG

# Bench discipline — same rounds/iters as queue_early_exit_headline.sh.
export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-10}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-200}

# Single-GPU only per CLAUDE.md memory (single_gpu_testing).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 4 primary shapes (B,T,H,KV,D). Matches queue_early_exit_headline.sh.
export WHALE_SHAPES="2,8192,8,4,64;2,4096,8,4,64;2,2048,8,4,64;1,16384,8,4,64"

# Optional: surface autotune picks in the log.
export TRITON_PRINT_AUTOTUNING=${TRITON_PRINT_AUTOTUNING:-1}
