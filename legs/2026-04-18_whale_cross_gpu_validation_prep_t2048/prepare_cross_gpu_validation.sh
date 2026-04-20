#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${LEG_DIR}/state"
LOG_DIR="${LEG_DIR}/logs"
EVID_DIR="${LEG_DIR}/evidence"
TASKS_FILE="${STATE_DIR}/tasks.txt"
MANIFEST_FILE="${STATE_DIR}/manifest.txt"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${EVID_DIR}"
source "${LEG_DIR}/tracked_env.sh"

: > "${TASKS_FILE}"
cat > "${MANIFEST_FILE}" <<EOF_MANIFEST
bench_shape=${BENCH_SHAPE}
bench_backends=${BENCH_BACKENDS}
probe_rounds=${BENCH_PROBE_ROUNDS}
probe_iters=${BENCH_PROBE_ITERS}
validate_rounds=${BENCH_VALIDATE_ROUNDS}
validate_iters=${BENCH_VALIDATE_ITERS}
anchor_file=${CROSS_GPU_ANCHOR_FILE}
anchor_status=${CROSS_GPU_WINNER_STATUS}
anchor_family=${CROSS_GPU_WINNER_FAMILY_ID}
anchor_dq_tile=${CROSS_GPU_WINNER_DQ_TILE}
anchor_dkdv_tile=${CROSS_GPU_WINNER_DKDV_TILE}
anchor_dkdv_mr=${CROSS_GPU_WINNER_DKDV_MAXNREG}
anchor_qmr=${CROSS_GPU_WINNER_QMR}
anchor_source=${CROSS_GPU_WINNER_SOURCE_EVIDENCE}
anchor_multi_gpu_ready=${CROSS_GPU_WINNER_MULTI_GPU_READY}
anchor_whale_donor_ready=${CROSS_GPU_WINNER_DONOR_READY}
anchor_qmrs=${CROSS_GPU_ANCHOR_QMRS}
challenger_rows=$(printf '%q' "${CROSS_GPU_CHALLENGER_ROWS}")
EOF_MANIFEST

append_task() {
  printf '%s\n' "$1" >> "${TASKS_FILE}"
}

for qmr in ${CROSS_GPU_ANCHOR_QMRS}; do
  append_task "${CROSS_GPU_WINNER_FAMILY_ID}|probe|${CROSS_GPU_WINNER_DQ_TILE}|${CROSS_GPU_WINNER_DKDV_TILE}|${CROSS_GPU_WINNER_DKDV_MAXNREG}|${qmr}"
done

while IFS='|' read -r family dq_tile dkdv_tile dkdv_mr qmrs; do
  [[ -n "${family}" ]] || continue
  for qmr in ${qmrs}; do
    append_task "${family}|probe|${dq_tile}|${dkdv_tile}|${dkdv_mr}|${qmr}"
  done
done <<< "${CROSS_GPU_CHALLENGER_ROWS}"

echo "prepared_tasks=$(wc -l < "${TASKS_FILE}")"
echo "manifest=${MANIFEST_FILE}"
echo "tasks=${TASKS_FILE}"
