#!/usr/bin/env bash
# Chained bench queue. Runs back-to-back on GPU 0 after the current
# early-exit bench (PID arg) finishes. Order:
#   1. Lever B  — fused_bwd_tma_dq variant on dkdv long-T shapes
#   2. Lever C  — fwd TMA long-T sweep (Phase A, patch-free)
# All benches serialize on GPU 0 per the single-GPU competition rule.
set -uo pipefail

WAIT_PID="${1:-}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
QUEUE_DIR="${REPO_ROOT}/legs/2026-04-16_whale_bench_queue"
mkdir -p "${QUEUE_DIR}/logs"
QLOG="${QUEUE_DIR}/logs/queue_$(date +%Y%m%d_%H%M%S).log"

PY=/venv/main/bin/python3
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0

echo "[queue] start ts=$(date +%FT%T) wait_pid=${WAIT_PID}" | tee "${QLOG}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[queue] waiting on PID ${WAIT_PID}" | tee -a "${QLOG}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 5; done
  echo "[queue] PID ${WAIT_PID} finished at $(date +%FT%T)" | tee -a "${QLOG}"
fi

# ------- Lever B: fused_bwd_tma_dq -------
LB_DIR="${REPO_ROOT}/legs/2026-04-16_whale_fused_bwd_tma_dq"
mkdir -p "${LB_DIR}/evidence" "${LB_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LB_LOG="${LB_DIR}/logs/queue_${TS}.log"
echo "[queue] === Lever B fused_bwd_tma_dq sweep ===" | tee -a "${QLOG}"
rm -rf /root/.triton/cache 2>/dev/null || true
export WHALE_BWD_VARIANT=fused_bwd_tma_dq
for shape in "2,8192,8,4,64" "2,4096,8,4,64" "2,2048,8,4,64" "1,16384,8,4,64"; do
    SAFE=$(echo "${shape}" | tr ',' '_')
    OUT="${LB_DIR}/evidence/lb_${SAFE}_${TS}.json"
    echo "[queue] LB shape=${shape} -> ${OUT}" | tee -a "${QLOG}"
    ${PY} "${REPO_ROOT}/legs/2026-04-16_whale_long_t_profile/bench_stable.py" \
        --shape "${shape}" \
        --backends "whale_fast,fa3" \
        --rounds 10 --iters 200 \
        --label "lb_${SAFE}" \
        --out "${OUT}" 2>&1 | tee -a "${LB_LOG}" | tee -a "${QLOG}" \
        || echo "[queue] LB shape=${shape} FAILED" | tee -a "${QLOG}"
done
unset WHALE_BWD_VARIANT

# ------- Lever C Phase A: fwd TMA long-T -------
LC_DIR="${REPO_ROOT}/legs/2026-04-16_whale_fwd_long_t"
mkdir -p "${LC_DIR}/evidence" "${LC_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LC_LOG="${LC_DIR}/logs/queue_${TS}.log"
echo "[queue] === Lever C Phase A fwd TMA long-T ===" | tee -a "${QLOG}"
export WHALE_FWD_VARIANT=tma
for shape in "2,4096,8,4,64" "2,8192,8,4,64" "1,16384,8,4,64"; do
    SAFE=$(echo "${shape}" | tr ',' '_')
    OUT="${LC_DIR}/evidence/lc_phaseA_${SAFE}_${TS}.json"
    echo "[queue] LC shape=${shape} -> ${OUT}" | tee -a "${QLOG}"
    rm -rf /root/.triton/cache 2>/dev/null || true
    ${PY} "${REPO_ROOT}/legs/2026-04-16_whale_long_t_profile/bench_stable.py" \
        --shape "${shape}" \
        --backends "whale_fast,fa3" \
        --rounds 10 --iters 200 \
        --label "lc_phaseA_${SAFE}" \
        --out "${OUT}" 2>&1 | tee -a "${LC_LOG}" | tee -a "${QLOG}" \
        || echo "[queue] LC shape=${shape} FAILED" | tee -a "${QLOG}"
done
unset WHALE_FWD_VARIANT

echo "[queue] all done ts=$(date +%FT%T)" | tee -a "${QLOG}"
