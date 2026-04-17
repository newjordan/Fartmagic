#!/usr/bin/env bash
# whale_longt_configs — Lever D (H10) driver.
# Bench whale_fast vs FA3 vs SDPA at the four current long-T loss shapes,
# both with and without WHALE_LONGT_CONFIGS=1.
#
# Shapes (the current losses from H6):
#   4,4096,8,4,64      — long-T GQA
#   2,8192,8,4,64      — very long-T GQA
#   4,2048,16,16,64    — wide-head MHA
#   2,2048,8,4,128     — hdim=128 GQA

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

SHAPES="4,4096,8,4,64;2,8192,8,4,64;4,2048,16,16,64;2,2048,8,4,128"
BACKENDS="whale_fast,fa3,sdpa"

echo "== ${TS}  Lever D long-T config sweep" | tee "${LOG}"
echo "== shapes=${SHAPES}" | tee -a "${LOG}"
echo "== backends=${BACKENDS}" | tee -a "${LOG}"

# A. Baseline: legacy config lists.
rm -rf /root/.triton/cache 2>/dev/null || true
WHALE_LONGT_CONFIGS=0 ${PY} "${LEG_DIR}/bench_stable.py" \
    --shapes "${SHAPES}" \
    --backends "${BACKENDS}" \
    --rounds "${WHALE_BENCH_ROUNDS}" \
    --iters  "${WHALE_BENCH_ITERS}" \
    --label "longt_configs_off" \
    --out "${LEG_DIR}/evidence/longt_configs_off_${TS}.json" 2>&1 | tee -a "${LOG}"

# B. Lever D: long-T-biased config lists.
rm -rf /root/.triton/cache 2>/dev/null || true
WHALE_LONGT_CONFIGS=1 ${PY} "${LEG_DIR}/bench_stable.py" \
    --shapes "${SHAPES}" \
    --backends "${BACKENDS}" \
    --rounds "${WHALE_BENCH_ROUNDS}" \
    --iters  "${WHALE_BENCH_ITERS}" \
    --label "longt_configs_on" \
    --out "${LEG_DIR}/evidence/longt_configs_on_${TS}.json" 2>&1 | tee -a "${LOG}"

echo "== done. log: ${LOG}"
echo "== evidence: ${LEG_DIR}/evidence/longt_configs_{off,on}_${TS}.json"
