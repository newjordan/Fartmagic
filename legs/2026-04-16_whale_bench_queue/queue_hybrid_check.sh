#!/usr/bin/env bash
# Head-to-head: whale_fast vs whale_hybrid (whale fwd + fa3 bwd) vs fa3.
# Chained after ee_precision PID arg. High precision (rounds=40 iters=800)
# on all four headline shapes.
set -uo pipefail

WAIT_PID="${1:-}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LEG_DIR="${REPO_ROOT}/legs/2026-04-17_whale_hybrid_headline"
mkdir -p "${LEG_DIR}/evidence" "${LEG_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
QLOG="${LEG_DIR}/logs/queue_${TS}.log"

PY=/venv/main/bin/python3
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export WHALE_BWD_VARIANT=baseline

BENCH="${REPO_ROOT}/legs/2026-04-16_whale_long_t_profile/bench_stable.py"

echo "[hybrid-check] start ts=$(date +%FT%T) wait_pid=${WAIT_PID}" | tee "${QLOG}"
if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 5; done
  echo "[hybrid-check] PID ${WAIT_PID} finished at $(date +%FT%T)" | tee -a "${QLOG}"
fi

for shape in "2,8192,8,4,64" "1,16384,8,4,64" "2,4096,8,4,64" "2,2048,8,4,64"; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    SAFE=$(echo "${shape}" | tr ',' '_')
    OUT="${LEG_DIR}/evidence/hybrid_${SAFE}_${TS}.json"
    echo "[hybrid-check] shape=${shape} -> ${OUT}" | tee -a "${QLOG}"
    ${PY} "${BENCH}" \
        --shape "${shape}" \
        --backends "whale_fast,whale_hybrid,fa3" \
        --rounds 40 --iters 800 \
        --label "hybrid_${SAFE}" \
        --out "${OUT}" 2>&1 | tee -a "${QLOG}" \
        || echo "[hybrid-check] shape=${shape} FAILED" | tee -a "${QLOG}"
done
echo "[hybrid-check] done ts=$(date +%FT%T)" | tee -a "${QLOG}"
