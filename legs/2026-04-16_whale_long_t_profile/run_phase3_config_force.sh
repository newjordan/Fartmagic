#!/usr/bin/env bash
# whale_long_t_profile — Phase 3 stop-gap.
# Force specific BLOCK_M,BLOCK_N,warps,stages on the dkdv baseline kernel
# at T=8192 and bench. Tests whether autotune is picking a poor config
# vs whether the search space itself is the problem.

set -euo pipefail

LEG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LEG_DIR}/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
source "${LEG_DIR}/tracked_env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY=/venv/main/bin/python3
export WHALE_BWD_VARIANT=baseline   # baseline at long T per Phase-1

LOG="${LEG_DIR}/logs/phase3_${TS}.log"
mkdir -p "${LEG_DIR}/logs" "${LEG_DIR}/evidence"
echo "== ${TS} Phase 3: force-config dkdv sweep at T=8192 (baseline variant)" | tee "${LOG}"

# Forced configs to test on _attn_bwd_dkdv_kernel.
# Format: "BM,BN,warps,stages"
CONFIGS=(
    "128,128,8,2"   # FA3's hdim64 choice
    "128,128,8,3"   # more stages
    "256,128,8,2"   # wider M
    "256,128,8,3"
    "128,256,8,2"   # wider N
    "128,256,8,3"
    "64,128,4,4"    # close to typical autotune pick
)

for CFG in "${CONFIGS[@]}"; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    echo "" | tee -a "${LOG}"
    echo "==== WHALE_BWD_KV_CONFIG=${CFG}" | tee -a "${LOG}"
    SAFE=$(echo "${CFG}" | tr ',' '_')
    WHALE_BWD_KV_CONFIG="${CFG}" ${PY} "${LEG_DIR}/bench_stable.py" \
        --shape "2,8192,8,4,64" \
        --backends "whale_fast,fa3" \
        --rounds 10 --iters 200 \
        --label "p3_kv_${SAFE}" \
        --out "${LEG_DIR}/evidence/p3_kv_${SAFE}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

echo "" | tee -a "${LOG}"
echo "== Phase 3 done. log: ${LOG}"
