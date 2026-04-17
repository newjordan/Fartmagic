#!/usr/bin/env bash
# Headline bench for the early-exit patch (dkdv baseline kernel).
# Waits for arg PID, then benches WHALE_BWD_VARIANT=baseline at the same
# 4 shapes as Lever A, using bench_stable.py (fwd+bwd headline numbers).
set -uo pipefail

WAIT_PID="${1:-}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
EE_DIR="${REPO_ROOT}/legs/2026-04-16_whale_dkdv_early_exit"
mkdir -p "${EE_DIR}/evidence" "${EE_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
QLOG="${EE_DIR}/logs/headline_${TS}.log"

PY=/venv/main/bin/python3
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export WHALE_BWD_VARIANT=baseline

echo "[ee-headline] start ts=$(date +%FT%T) wait_pid=${WAIT_PID}" | tee "${QLOG}"
if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 5; done
  echo "[ee-headline] PID ${WAIT_PID} finished at $(date +%FT%T)" | tee -a "${QLOG}"
fi

rm -rf /root/.triton/cache 2>/dev/null || true
for shape in "2,8192,8,4,64" "2,4096,8,4,64" "2,2048,8,4,64" "1,16384,8,4,64"; do
    SAFE=$(echo "${shape}" | tr ',' '_')
    OUT="${EE_DIR}/evidence/ee_headline_${SAFE}_${TS}.json"
    echo "[ee-headline] shape=${shape} -> ${OUT}" | tee -a "${QLOG}"
    ${PY} "${REPO_ROOT}/legs/2026-04-16_whale_long_t_profile/bench_stable.py" \
        --shape "${shape}" \
        --backends "whale_fast,fa3" \
        --rounds 10 --iters 200 \
        --label "ee_${SAFE}" \
        --out "${OUT}" 2>&1 | tee -a "${QLOG}" \
        || echo "[ee-headline] shape=${shape} FAILED" | tee -a "${QLOG}"
done
echo "[ee-headline] done ts=$(date +%FT%T)" | tee -a "${QLOG}"
