#!/usr/bin/env bash
# Fwd micro-ablations: maxnreg={128,224} + num_stages=6 expanded autotune.
# 48->74 configs. Pure whale_fast vs FA3 only. User owns hybrid lane.
set -uo pipefail
REPO_ROOT=/workspace/SOTA_FINAL
LEG_DIR="${REPO_ROOT}/legs/2026-04-17_whale_fwd_micro"
mkdir -p "${LEG_DIR}/evidence" "${LEG_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
QLOG="${LEG_DIR}/logs/queue_${TS}.log"
PY=/venv/main/bin/python3
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export WHALE_BWD_VARIANT=baseline
BENCH="${REPO_ROOT}/legs/2026-04-16_whale_long_t_profile/bench_stable.py"

${PY} -c "from vault.whale_kernel_triton import _fwd_configs; n=len(_fwd_configs()); print('_fwd_configs=',n); assert n==74,'patch not live: '+str(n)" 2>&1 | tee "${QLOG}"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then echo "[fwd-micro] FATAL patch not live" | tee -a "${QLOG}"; exit 2; fi

echo "[fwd-micro] start ts=$(date +%FT%T)" | tee -a "${QLOG}"

# Phase 1: T sweep at B=2 H=8 KV=4 D=64
for T in 2048 4096 8192; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    SHAPE="2,${T},8,4,64"
    SAFE=$(echo "${SHAPE}" | tr ',' '_')
    OUT="${LEG_DIR}/evidence/micro_${SAFE}_${TS}.json"
    echo "[fwd-micro] shape=${SHAPE}" | tee -a "${QLOG}"
    ${PY} "${BENCH}" --shape "${SHAPE}" --backends "whale_fast,fa3" \
        --rounds 30 --iters 600 --label "micro_${SAFE}" --out "${OUT}" 2>&1 | tee -a "${QLOG}" \
        || echo "[fwd-micro] FAILED ${SHAPE}" | tee -a "${QLOG}"
done

# Phase 2: capture autotune winners
for T in 2048 8192; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    SHAPE="2,${T},8,4,64"
    SAFE=$(echo "${SHAPE}" | tr ',' '_')
    AOUT="${LEG_DIR}/evidence/winner_${SAFE}_${TS}.log"
    echo "[fwd-micro] winner-capture shape=${SHAPE}" | tee -a "${QLOG}"
    TRITON_PRINT_AUTOTUNING=1 ${PY} "${BENCH}" --shape "${SHAPE}" --backends "whale_fast" \
        --rounds 1 --iters 1 --label "winner_${SAFE}" --out /tmp/winner_${SAFE}.json 2>&1 \
        | grep -iE "_attn_fwd_kernel|BLOCK_M|num_warps|maxnreg" | head -40 | tee "${AOUT}"
done

echo "[fwd-micro] done ts=$(date +%FT%T)" | tee -a "${QLOG}"
