#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LEG_DIR}/logs"
STATE_DIR="${LEG_DIR}/state"

source "${LEG_DIR}/tracked_env.sh"

echo "---ANCHOR---"
jq . "${CROSS_GPU_ANCHOR_FILE}"
echo "---GPU---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits
echo "---PIDS---"
ps -ef | grep -E 'run_cross_gpu_worker|launch_cross_gpu_validation' | grep -v grep || true
echo "---LAUNCH---"
tail -n 80 "${LOG_DIR}/cross_gpu_launch.log" 2>/dev/null || true
echo "---WORKERS---"
for f in "${LOG_DIR}"/worker_*.log; do
  [[ -f "${f}" ]] || continue
  echo "### $(basename "${f}")"
  tail -n 40 "${f}"
done
echo "---QUEUE---"
wc -l "${STATE_DIR}/tasks.txt" 2>/dev/null || true
echo "---EVIDENCE---"
find "${LEG_DIR}/evidence" -maxdepth 1 -type f | wc -l
