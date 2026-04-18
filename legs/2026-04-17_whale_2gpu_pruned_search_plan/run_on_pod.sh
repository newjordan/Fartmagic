#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LEG_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

for worker_id in 0 1; do
  nohup bash "${LEG_DIR}/run_worker_slice.sh" "${worker_id}" "${worker_id}" > "${LOG_DIR}/driver_worker${worker_id}_${TS}.log" 2>&1 &
  echo "worker${worker_id}_pid=$!" | tee -a "${LOG_DIR}/launch_${TS}.log"
done

echo "launch_ts=${TS}" | tee -a "${LOG_DIR}/launch_${TS}.log"
