#!/usr/bin/env bash
# whale_bwd_tma_dkdv — H2 benchmark driver.
# Numerics check, then headline head-to-head fused_delta vs fused_delta_tma vs FA3.

set -euo pipefail

LEG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LEG_DIR}/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
source "${LEG_DIR}/tracked_env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY=/venv/main/bin/python3

LOG="${LEG_DIR}/logs/run_${TS}.log"
mkdir -p "${LEG_DIR}/logs" "${LEG_DIR}/evidence"
echo "== ${TS}  WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT}" | tee "${LOG}"

# 1. Numerics check vs SDPA reference (fused_delta_tma).
rm -rf /root/.triton/cache 2>/dev/null || true
WHALE_BWD_VARIANT=fused_delta_tma ${PY} legs/2026-04-16_whale_fast_kernels/bench_numerics.py \
    --kernel vault.whale_kernel_triton:whale_attn_fast \
    --label fused_delta_tma_numerics \
    --out "${LEG_DIR}/evidence/fused_delta_tma_numerics_${TS}.json" 2>&1 | tee -a "${LOG}"

# 2. Headline head-to-head: fused_delta_tma vs fused_delta vs fa3 vs sdpa.
for VARIANT in fused_delta fused_delta_tma; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    WHALE_BWD_VARIANT=${VARIANT} ${PY} "${LEG_DIR}/bench_stable.py" \
        --shape "4,2048,8,4,64" \
        --backends "whale_fast,fa3,sdpa" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label "headline_${VARIANT}" \
        --out "${LEG_DIR}/evidence/headline_${VARIANT}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

# 3. Shape sweep (only D∈{32,64,128} — TMA requires BLOCK_D == D).
for VARIANT in fused_delta fused_delta_tma; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    WHALE_BWD_VARIANT=${VARIANT} ${PY} "${LEG_DIR}/bench_stable.py" \
        --shapes "4,1024,8,4,64;4,2048,8,4,64;4,4096,8,4,64;2,8192,8,4,64;4,2048,8,8,64;4,2048,16,16,64;2,2048,8,4,128;4,2048,8,4,32" \
        --backends "whale_fast,fa3,sdpa" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label "sweep_${VARIANT}" \
        --out "${LEG_DIR}/evidence/sweep_${VARIANT}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

echo "== done. log: ${LOG}"
