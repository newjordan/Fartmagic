#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${LEG_DIR}/state"
LOG_DIR="${LEG_DIR}/logs"
TASKS_FILE="${STATE_DIR}/tasks.txt"
DONE_FILE="${STATE_DIR}/done.txt"
LAUNCH_LOG="${LOG_DIR}/cross_gpu_launch.log"

mkdir -p "${STATE_DIR}" "${LOG_DIR}"
source "${LEG_DIR}/tracked_env.sh"

if [[ ! -s "${TASKS_FILE}" ]]; then
  bash "${LEG_DIR}/prepare_cross_gpu_validation.sh" >> "${LAUNCH_LOG}" 2>&1
fi

: > "${DONE_FILE}"

log() {
  echo "[$(date +%F_%T)] $*" | tee -a "${LAUNCH_LOG}"
}

log "launch_start workers=${CROSS_GPU_WORKER_GPUS}"
for gpu in ${CROSS_GPU_WORKER_GPUS}; do
  nohup bash "${LEG_DIR}/run_cross_gpu_worker.sh" "${gpu}" > "${LOG_DIR}/worker_${gpu}.log" 2>&1 &
  log "worker_${gpu}_pid=$!"
done
log "launch_done"
