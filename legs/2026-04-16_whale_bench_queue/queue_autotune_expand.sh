#!/usr/bin/env bash
# Chained bench: runs autotune-expand sweep (Lever E) after the current
# ee_headline bench (PID arg) finishes. 4 shapes, whale_fast+fa3, with
# /root/.triton/cache flushed per shape to force cold autotune over the
# expanded 56-config space.
set -uo pipefail

WAIT_PID="${1:-}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LEG_DIR="${REPO_ROOT}/legs/2026-04-17_whale_dkdv_autotune_expand"
mkdir -p "${LEG_DIR}/evidence" "${LEG_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
QLOG="${LEG_DIR}/logs/queue_${TS}.log"

PY=/venv/main/bin/python3
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export WHALE_BWD_VARIANT=baseline

echo "[autotune-expand] start ts=$(date +%FT%T) wait_pid=${WAIT_PID}" | tee "${QLOG}"
if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 5; done
  echo "[autotune-expand] PID ${WAIT_PID} finished at $(date +%FT%T)" | tee -a "${QLOG}"
fi

# Assert 56 configs to confirm vault patch is live.
${PY} -c "from vault.whale_kernel_triton import _bwd_kv_configs; n=len(_bwd_kv_configs()); print('_bwd_kv_configs =', n); assert n == 56, 'expected 56, got '+str(n)" 2>&1 | tee -a "${QLOG}"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "[autotune-expand] FATAL: vault_patch not live" | tee -a "${QLOG}"
    exit 2
fi

BENCH="${REPO_ROOT}/legs/2026-04-16_whale_long_t_profile/bench_stable.py"
for shape in "2,8192,8,4,64" "2,4096,8,4,64" "2,2048,8,4,64" "1,16384,8,4,64"; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    SAFE=$(echo "${shape}" | tr ',' '_')
    OUT="${LEG_DIR}/evidence/expand_${SAFE}_${TS}.json"
    echo "[autotune-expand] shape=${shape} -> ${OUT}" | tee -a "${QLOG}"
    ${PY} "${BENCH}" \
        --shape "${shape}" \
        --backends "whale_fast,fa3" \
        --rounds 10 --iters 200 \
        --label "expand_${SAFE}" \
        --out "${OUT}" 2>&1 | tee -a "${QLOG}" \
        || echo "[autotune-expand] shape=${shape} FAILED" | tee -a "${QLOG}"
done
echo "[autotune-expand] done ts=$(date +%FT%T)" | tee -a "${QLOG}"
