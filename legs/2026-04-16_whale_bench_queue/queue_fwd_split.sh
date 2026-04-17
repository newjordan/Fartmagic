#!/usr/bin/env bash
# Causal-split fwd: phase-1 fully-unmasked tiles (no tl.where, no T-mask),
# phase-2 boundary tiles. Stacked on top of maxnreg/s=6 ablation (74 configs).
set -uo pipefail
REPO_ROOT=/workspace/SOTA_FINAL
LEG_DIR="${REPO_ROOT}/legs/2026-04-17_whale_fwd_causal_split"
mkdir -p "${LEG_DIR}/evidence" "${LEG_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
QLOG="${LEG_DIR}/logs/queue_${TS}.log"
PY=/venv/main/bin/python3
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export WHALE_BWD_VARIANT=baseline
BENCH="${REPO_ROOT}/legs/2026-04-16_whale_long_t_profile/bench_stable.py"

${PY} -c "from vault.whale_kernel_triton import _fwd_configs; print('_fwd_configs=',len(_fwd_configs()))" 2>&1 | tee "${QLOG}"

echo "[fwd-split] start ts=$(date +%FT%T)" | tee -a "${QLOG}"

for shape in "2,2048,8,4,64" "2,4096,8,4,64" "2,8192,8,4,64"; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    SAFE=$(echo "${shape}" | tr ',' '_')
    OUT="${LEG_DIR}/evidence/split_${SAFE}_${TS}.json"
    echo "[fwd-split] shape=${shape}" | tee -a "${QLOG}"
    ${PY} "${BENCH}" --shape "${shape}" --backends "whale_fast,fa3" \
        --rounds 30 --iters 600 --label "split_${SAFE}" --out "${OUT}" 2>&1 | tee -a "${QLOG}" \
        || echo "[fwd-split] FAILED ${shape}" | tee -a "${QLOG}"
done

echo "[fwd-split] done ts=$(date +%FT%T)" | tee -a "${QLOG}"
