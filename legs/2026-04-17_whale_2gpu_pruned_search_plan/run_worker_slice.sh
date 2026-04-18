#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_ID="${1:?worker id required}"
GPU_ID="${2:-0}"
PY="/venv/main/bin/python3"
LOG_DIR="${LEG_DIR}/logs"
EVID_DIR="${LEG_DIR}/evidence"
TS="$(date +%Y%m%d_%H%M%S)"
WORKER_LOG="${LOG_DIR}/worker${WORKER_ID}_${TS}.log"
CACHE_DIR="/tmp/triton_cache_whale_worker${WORKER_ID}"
WORKER_DIR="${LEG_DIR}/runtime/worker${WORKER_ID}"

mkdir -p "${LOG_DIR}" "${EVID_DIR}" "${WORKER_DIR}/logs" "${WORKER_DIR}/evidence" "${CACHE_DIR}"
source "${LEG_DIR}/tracked_env.sh"

getv() {
  local name="$1"
  printf '%s' "${!name}"
}

MODE="$(getv WORKER${WORKER_ID}_MODE)"
cp "${LEG_DIR}/bench_stable_scratch.py" "${WORKER_DIR}/bench_stable_scratch.py"
mkdir -p "${WORKER_DIR}/frozen_vault"
cp "${LEG_DIR}/frozen_vault/__init__.py" "${WORKER_DIR}/frozen_vault/__init__.py"
cp "${LEG_DIR}/frozen_vault/whale_kernel_triton.py" "${WORKER_DIR}/frozen_vault/whale_kernel_triton.py"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TRITON_CACHE_DIR="${CACHE_DIR}"
export PYTHONPATH="${WORKER_DIR}:${PYTHONPATH:-}"
export WHALE_BWD_VARIANT
export WHALE_FUSED_DELTA_T_MAX

echo "[$(date +%F_%T)] worker=${WORKER_ID} gpu=${GPU_ID} mode=${MODE}" | tee -a "${WORKER_LOG}"
echo "shape=${BENCH_SHAPE} backends=${BENCH_BACKENDS} rounds=${BENCH_ROUNDS} iters=${BENCH_ITERS}" | tee -a "${WORKER_LOG}"
echo "scratch_source_commit=${SCRATCH_KERNEL_SOURCE_COMMIT}" | tee -a "${WORKER_LOG}"
echo "scratch_kernel_sha256=$(sha256sum "${WORKER_DIR}/frozen_vault/whale_kernel_triton.py" | awk '{print $1}')" | tee -a "${WORKER_LOG}"

run_case() {
  local case_label="$1"
  local dkdv_tile="$2"
  local dkdv_mr="$3"
  local dq_tile="$4"
  local dq_mr="$5"
  local safe_label="${case_label//,/_}"
  local out="${EVID_DIR}/worker${WORKER_ID}_${safe_label}_${TS}.json"
  local runlog="${WORKER_DIR}/logs/${safe_label}_${TS}.log"

  export WHALE_BWD_KV_TMA_CONFIG="${dkdv_tile}"
  export WHALE_BWD_KV_TMA_MAXNREG="${dkdv_mr}"
  export WHALE_BWD_Q_CONFIG="${dq_tile}"
  export WHALE_BWD_Q_MAXNREG="${dq_mr}"

  echo "[$(date +%T)] case=${case_label} start" | tee -a "${WORKER_LOG}"
  rm -rf "${CACHE_DIR}"/* 2>/dev/null || true
  if "${PY}" "${WORKER_DIR}/bench_stable_scratch.py" \
      --label "worker${WORKER_ID}_${safe_label}" \
      --shape "${BENCH_SHAPE}" \
      --backends "${BENCH_BACKENDS}" \
      --rounds "${BENCH_ROUNDS}" \
      --iters "${BENCH_ITERS}" \
      --out "${out}" >>"${runlog}" 2>&1; then
    tail -n 5 "${runlog}" | tee -a "${WORKER_LOG}"
    echo "[$(date +%T)] case=${case_label} done" | tee -a "${WORKER_LOG}"
  else
    echo "[$(date +%T)] case=${case_label} FAIL (see ${runlog})" | tee -a "${WORKER_LOG}"
  fi
}

case "${MODE}" in
  dq_cartesian)
    DKDV_TILE="$(getv WORKER${WORKER_ID}_DKDV_TILE)"
    DKDV_MR="$(getv WORKER${WORKER_ID}_DKDV_MAXNREG)"
    for DQ_TILE in $(getv WORKER${WORKER_ID}_DQ_TILES); do
      for DQ_MR in $(getv WORKER${WORKER_ID}_DQ_MAXNREGS); do
        run_case "dqtile_${DQ_TILE}_dqmr_${DQ_MR}_dkdvmr_${DKDV_MR}" "${DKDV_TILE}" "${DKDV_MR}" "${DQ_TILE}" "${DQ_MR}"
      done
    done
    ;;
  dkdv_maxnreg)
    DKDV_TILE="$(getv WORKER${WORKER_ID}_DKDV_TILE)"
    DQ_TILE="$(getv WORKER${WORKER_ID}_DQ_TILE)"
    DQ_MR="$(getv WORKER${WORKER_ID}_DQ_MAXNREG)"
    for DKDV_MR in $(getv WORKER${WORKER_ID}_DKDV_MAXNREGS); do
      run_case "dkdvmr_${DKDV_MR}_dqtile_${DQ_TILE}_dqmr_${DQ_MR}" "${DKDV_TILE}" "${DKDV_MR}" "${DQ_TILE}" "${DQ_MR}"
    done
    ;;
  dkdv_tile)
    DKDV_MR="$(getv WORKER${WORKER_ID}_DKDV_MAXNREG)"
    DQ_TILE="$(getv WORKER${WORKER_ID}_DQ_TILE)"
    DQ_MR="$(getv WORKER${WORKER_ID}_DQ_MAXNREG)"
    for DKDV_TILE in $(getv WORKER${WORKER_ID}_DKDV_TILES); do
      run_case "dkdvtile_${DKDV_TILE}_dkdvmr_${DKDV_MR}_dqtile_${DQ_TILE}_dqmr_${DQ_MR}" "${DKDV_TILE}" "${DKDV_MR}" "${DQ_TILE}" "${DQ_MR}"
    done
    ;;
  *)
    echo "unknown mode=${MODE}" >&2
    exit 1
    ;;
esac

echo "[$(date +%F_%T)] worker=${WORKER_ID} done" | tee -a "${WORKER_LOG}"
