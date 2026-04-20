#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="${1:?family required}"
PHASE="${2:?phase required}"
DQ_TILE="${3:?dq tile required}"
DKDV_TILE="${4:?dkdv tile required}"
DKDV_MR="${5:?dkdv maxnreg required}"
QMR="${6:?qmr required}"
GPU_ID="${7:?gpu id required}"
TS="$(date +%Y%m%d_%H%M%S)"
SAFE_DQ_TILE="${DQ_TILE//,/_}"
SAFE_DKDV_TILE="${DKDV_TILE//,/_}"
LABEL="cross_${FAMILY}_${PHASE}_dq_${SAFE_DQ_TILE}_dkdv_${SAFE_DKDV_TILE}_mr_${DKDV_MR}_qmr_${QMR}"
LOG_DIR="${LEG_DIR}/logs"
EVID_DIR="${LEG_DIR}/evidence"
CACHE_DIR="${CROSS_GPU_CACHE_ROOT:-/tmp/triton_cache_whale_cross_gpu}"
WORK_CACHE="${CACHE_DIR}/${LABEL}_${GPU_ID}_${TS}"
OUT="${EVID_DIR}/${LABEL}_${TS}.json"
RUNLOG="${LOG_DIR}/${LABEL}_${TS}.log"
PY="/venv/main/bin/python3"

mkdir -p "${LOG_DIR}" "${EVID_DIR}" "${WORK_CACHE}/triton" "${WORK_CACHE}/inductor" "${WORK_CACHE}/xdg" "${WORK_CACHE}/tmp"
source "${LEG_DIR}/tracked_env.sh"

case "${PHASE}" in
  probe)
    ROUNDS="${BENCH_PROBE_ROUNDS}"
    ITERS="${BENCH_PROBE_ITERS}"
    ;;
  validate)
    ROUNDS="${BENCH_VALIDATE_ROUNDS}"
    ITERS="${BENCH_VALIDATE_ITERS}"
    ;;
  *)
    echo "unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac

DQ_MR="${QMR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TRITON_CACHE_DIR="${WORK_CACHE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${WORK_CACHE}/inductor"
export XDG_CACHE_HOME="${WORK_CACHE}/xdg"
export TMPDIR="${WORK_CACHE}/tmp"
export PYTHONPATH="/workspace/SOTA_FINAL:${PYTHONPATH:-}"
export WHALE_BWD_VARIANT
export WHALE_FUSED_DELTA_T_MAX
export BENCH_SHAPE
export BENCH_BACKENDS
export WHALE_BWD_KV_TMA_CONFIG="${DKDV_TILE}"
export WHALE_BWD_KV_TMA_MAXNREG="${DKDV_MR}"
export WHALE_BWD_Q_CONFIG="${DQ_TILE}"
export WHALE_BWD_Q_MAXNREG="${DQ_MR}"

echo "[$(date +%F_%T)] label=${LABEL} gpu=${GPU_ID} family=${FAMILY} phase=${PHASE}" | tee -a "${RUNLOG}"
echo "qmr=${QMR} rounds=${ROUNDS} iters=${ITERS}" | tee -a "${RUNLOG}"
echo "dq_tile=${DQ_TILE} dq_mr=${DQ_MR} dkdv_tile=${DKDV_TILE} dkdv_mr=${DKDV_MR}" | tee -a "${RUNLOG}"
echo "anchor_file=${CROSS_GPU_ANCHOR_FILE} anchor_status=${CROSS_GPU_WINNER_STATUS}" | tee -a "${RUNLOG}"
echo "triton_cache=${TRITON_CACHE_DIR}" | tee -a "${RUNLOG}"
echo "torchinductor_cache=${TORCHINDUCTOR_CACHE_DIR}" | tee -a "${RUNLOG}"

"${PY}" "${CROSS_GPU_BENCH_SCRIPT}" \
  --label "${LABEL}" \
  --shape "${BENCH_SHAPE}" \
  --backends "${BENCH_BACKENDS}" \
  --rounds "${ROUNDS}" \
  --iters "${ITERS}" \
  --out "${OUT}" >> "${RUNLOG}" 2>&1

tail -n 8 "${RUNLOG}"
