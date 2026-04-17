#!/usr/bin/env bash
# whale_long_t_profile — H6 driver.
# Phase 1: variant sweep at T=4096, T=8192.
# Phase 2: kineto trace whale-best vs FA3 at T=8192 (separate script).

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
echo "== ${TS}  WHALE_BWD_VARIANT phase-1 sweep" | tee "${LOG}"

# Phase 1: variant sweep at long T.
# Shapes: 4,4096,8,4,64 and 2,8192,8,4,64 (both showed >80% gap to FA3 in H5).
for VARIANT in auto baseline fused_delta; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    WHALE_BWD_VARIANT=${VARIANT} ${PY} "${LEG_DIR}/bench_stable.py" \
        --shapes "4,4096,8,4,64;2,8192,8,4,64" \
        --backends "whale_fast,fa3,sdpa" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label "longt_${VARIANT}" \
        --out "${LEG_DIR}/evidence/longt_${VARIANT}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

echo "== done. log: ${LOG}"
