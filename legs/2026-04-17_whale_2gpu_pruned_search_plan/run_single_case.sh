#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${1:?label required}"
GPU_ID="${2:?gpu id required}"
ROUNDS="${3:?rounds required}"
ITERS="${4:?iters required}"
DKDV_TILE="${5:?dkdv tile required}"
DKDV_MR="${6:?dkdv maxnreg required}"
DQ_TILE="${7:?dq tile required}"
DQ_MR="${8:?dq maxnreg required}"
PY="/venv/main/bin/python3"
TS="$(date +%Y%m%d_%H%M%S)"
SAFE_LABEL="${LABEL//,/_}"
LOG_DIR="${LEG_DIR}/logs"
EVID_DIR="${LEG_DIR}/evidence"
RUNTIME_DIR="${LEG_DIR}/runtime/${SAFE_LABEL}_${TS}"
CACHE_DIR="/tmp/triton_cache_${SAFE_LABEL}_${GPU_ID}_${TS}"
RUNLOG="${LOG_DIR}/${SAFE_LABEL}_${TS}.log"
OUT="${EVID_DIR}/${SAFE_LABEL}_${TS}.json"

mkdir -p "${LOG_DIR}" "${EVID_DIR}" "${RUNTIME_DIR}/frozen_vault" "${CACHE_DIR}"
source "${LEG_DIR}/tracked_env.sh"

cp "${LEG_DIR}/bench_stable_scratch.py" "${RUNTIME_DIR}/bench_stable_scratch.py"
cp "${LEG_DIR}/frozen_vault/__init__.py" "${RUNTIME_DIR}/frozen_vault/__init__.py"
cp "${LEG_DIR}/frozen_vault/whale_kernel_triton.py" "${RUNTIME_DIR}/frozen_vault/whale_kernel_triton.py"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TRITON_CACHE_DIR="${CACHE_DIR}"
export PYTHONPATH="${RUNTIME_DIR}:${PYTHONPATH:-}"
export WHALE_BWD_VARIANT
export WHALE_FUSED_DELTA_T_MAX
export WHALE_BWD_KV_TMA_CONFIG="${DKDV_TILE}"
export WHALE_BWD_KV_TMA_MAXNREG="${DKDV_MR}"
export WHALE_BWD_Q_CONFIG="${DQ_TILE}"
export WHALE_BWD_Q_MAXNREG="${DQ_MR}"

echo "[$(date +%F_%T)] label=${LABEL} gpu=${GPU_ID} rounds=${ROUNDS} iters=${ITERS}" | tee -a "${RUNLOG}"
echo "dkdv_tile=${DKDV_TILE} dkdv_mr=${DKDV_MR} dq_tile=${DQ_TILE} dq_mr=${DQ_MR}" | tee -a "${RUNLOG}"
echo "scratch_source_commit=${SCRATCH_KERNEL_SOURCE_COMMIT}" | tee -a "${RUNLOG}"
echo "scratch_kernel_sha256=$(sha256sum "${RUNTIME_DIR}/frozen_vault/whale_kernel_triton.py" | awk '{print $1}')" | tee -a "${RUNLOG}"

"${PY}" "${RUNTIME_DIR}/bench_stable_scratch.py" \
  --label "${LABEL}" \
  --shape "${BENCH_SHAPE}" \
  --backends "${BENCH_BACKENDS}" \
  --rounds "${ROUNDS}" \
  --iters "${ITERS}" \
  --out "${OUT}" >>"${RUNLOG}" 2>&1

tail -n 8 "${RUNLOG}"
