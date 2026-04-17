#!/usr/bin/env bash
# whale_bwd_persistent_atomic — H5 benchmark driver.
# Runs (1) numerics check vs SDPA, (2) headline bench, (3) full shape sweep.
# Uses fresh triton cache to avoid stale autotune picks.

set -euo pipefail

LEG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LEG_DIR}/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
source "${LEG_DIR}/tracked_env.sh"

# Fresh autotune cache so the new config list is actually picked.
rm -rf /root/.triton/cache 2>/dev/null || true

LOG="${LEG_DIR}/logs/run_${TS}.log"
mkdir -p "${LEG_DIR}/logs" "${LEG_DIR}/evidence"
echo "== ${TS}  WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT}" | tee "${LOG}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY=/venv/main/bin/python3

# 1. Numerics check vs SDPA reference across all headline shapes.
WHALE_BWD_VARIANT=fused_bwd ${PY} legs/2026-04-16_whale_fast_kernels/bench_numerics.py \
    --kernel vault.whale_kernel_triton:whale_attn_fast \
    --label fused_bwd_numerics \
    --out "${LEG_DIR}/evidence/fused_bwd_numerics_${TS}.json" 2>&1 | tee -a "${LOG}"

# 2. Headline head-to-head: fused_bwd vs fused_delta (current winner) vs fa3 vs sdpa.
#    Two passes (one per WHALE_BWD_VARIANT) so each gets its own warm autotune cache.
for VARIANT in fused_delta fused_bwd; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    WHALE_BWD_VARIANT=${VARIANT} ${PY} "${LEG_DIR}/bench_stable.py" \
        --shape "4,2048,8,4,64" \
        --backends "whale_fast,fa3,sdpa" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label "headline_${VARIANT}" \
        --out "${LEG_DIR}/evidence/headline_${VARIANT}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

# 3. Full definitive shape sweep, both variants.
for VARIANT in fused_delta fused_bwd; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    WHALE_BWD_VARIANT=${VARIANT} ${PY} "${LEG_DIR}/bench_stable.py" \
        --shapes "4,1024,8,4,64;4,2048,8,4,64;4,4096,8,4,64;2,8192,8,4,64;4,2048,8,8,64;4,2048,16,16,64;2,2048,8,4,128" \
        --backends "whale_fast,fa3,sdpa" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label "sweep_${VARIANT}" \
        --out "${LEG_DIR}/evidence/sweep_${VARIANT}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

echo "== done. log: ${LOG}"
