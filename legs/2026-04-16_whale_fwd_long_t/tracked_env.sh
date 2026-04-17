# whale_fwd_long_t — env for H9 (forward at long T).
# Source before run.sh.

export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="9.0"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.triton/cache}

# H9 baseline: TMA forward variant.
#   - run_phase_a.sh (Lever C Phase A): TMA-only, no forced config.
#   - run.sh Phase A: overrides per-call to also bench the default (non-TMA).
export WHALE_FWD_VARIANT=${WHALE_FWD_VARIANT:-tma}

# H9b knob — only takes effect once the patch in vault_patch.md is applied
# to vault/whale_kernel_triton.py. Default off so this env is safe to source
# even if the vault is unpatched.
export WHALE_FWD_TMA_WARPSPEC=${WHALE_FWD_TMA_WARPSPEC:-0}

# Bench shape for the inner loop. Long-T means we need fewer iters per round
# for stable timing; the headline T=8192 shape is heavy.
export WHALE_BENCH_ROUNDS=${WHALE_BENCH_ROUNDS:-15}
export WHALE_BENCH_ITERS=${WHALE_BENCH_ITERS:-300}

# Phase B candidate configs (FA3-like tiles), comma-joined as BM,BN,W,S.
# run.sh iterates over these via WHALE_FWD_TMA_CONFIG.
export WHALE_FWD_TMA_PHASEB_CONFIGS=${WHALE_FWD_TMA_PHASEB_CONFIGS:-"128,128,8,2 128,128,8,3 256,128,8,2 128,256,8,2"}
