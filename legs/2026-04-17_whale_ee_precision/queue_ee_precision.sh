#!/usr/bin/env bash
# High-precision a/b for the early-exit split on _attn_bwd_dkdv_kernel.
# Chain after PID arg. At each phase:
#   - swap vault to target form (split / unsplit)
#   - clear triton cache (force cold autotune)
#   - bench (2,8192,8,4,64) and (1,16384,8,4,64) at rounds=40 iters=800
#     -> ~5x tighter sigma vs the rounds=10 iters=200 default.
# Restores vault to "split" at the end.
set -uo pipefail

WAIT_PID="${1:-}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LEG_DIR="${REPO_ROOT}/legs/2026-04-17_whale_ee_precision"
mkdir -p "${LEG_DIR}/evidence" "${LEG_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
QLOG="${LEG_DIR}/logs/queue_${TS}.log"

PY=/venv/main/bin/python3
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export WHALE_BWD_VARIANT=baseline

BENCH="${REPO_ROOT}/legs/2026-04-16_whale_long_t_profile/bench_stable.py"
SWAP="${LEG_DIR}/swap_ee.py"

run_shape () {
    local form="$1" shape="$2"
    local safe
    safe=$(echo "${shape}" | tr ',' '_')
    local out="${LEG_DIR}/evidence/${form}_${safe}_${TS}.json"
    echo "[ee-precision] form=${form} shape=${shape} -> ${out}" | tee -a "${QLOG}"
    rm -rf /root/.triton/cache 2>/dev/null || true
    ${PY} "${BENCH}" \
        --shape "${shape}" \
        --backends "whale_fast,fa3" \
        --rounds 40 --iters 800 \
        --label "${form}_${safe}" \
        --out "${out}" 2>&1 | tee -a "${QLOG}"
}

echo "[ee-precision] start ts=$(date +%FT%T) wait_pid=${WAIT_PID}" | tee "${QLOG}"
if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 5; done
  echo "[ee-precision] PID ${WAIT_PID} finished at $(date +%FT%T)" | tee -a "${QLOG}"
fi

# Phase A: split (current vault, early-exit active)
${PY} "${SWAP}" split 2>&1 | tee -a "${QLOG}"
run_shape split "2,8192,8,4,64"
run_shape split "1,16384,8,4,64"

# Phase B: unsplit (pre-patch form, single M-loop w/ causal tl.where)
${PY} "${SWAP}" unsplit 2>&1 | tee -a "${QLOG}"
run_shape unsplit "2,8192,8,4,64"
run_shape unsplit "1,16384,8,4,64"

# Restore split (canonical vault state)
${PY} "${SWAP}" split 2>&1 | tee -a "${QLOG}"

echo "[ee-precision] done ts=$(date +%FT%T)" | tee -a "${QLOG}"
