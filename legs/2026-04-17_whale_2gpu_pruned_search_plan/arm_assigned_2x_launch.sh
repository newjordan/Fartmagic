#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:?set INSTANCE_ID}"
SSH_HOST="${SSH_HOST:?set SSH_HOST}"
SSH_PORT="${SSH_PORT:?set SSH_PORT}"
SSH_KEY="/home/frosty40/.ssh/id_ed25519_apollo"
VASTAI="${VASTAI:-/home/frosty40/miniconda3/bin/vastai}"
LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LEG_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/arm_assigned_2x_${TS}.log"
mkdir -p "${LOG_DIR}"

while true; do
  JSON="$("${VASTAI}" show instance "${INSTANCE_ID}" --raw)"
  ACTUAL="$(printf '%s' "${JSON}" | jq -r '.actual_status')"
  INTENDED="$(printf '%s' "${JSON}" | jq -r '.intended_status')"
  echo "[$(date +%F_%T)] actual=${ACTUAL} intended=${INTENDED}" | tee -a "${LOG}"

  if [[ "${ACTUAL}" != "running" || "${INTENDED}" != "running" ]]; then
    "${VASTAI}" start instance "${INSTANCE_ID}" | tee -a "${LOG}" || true
    sleep 30
    continue
  fi

  if ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=no -p "${SSH_PORT}" root@"${SSH_HOST}" 'echo SSH_OK' >/dev/null 2>&1; then
    echo "[$(date +%F_%T)] ssh_ready launching" | tee -a "${LOG}"
    bash "${LEG_DIR}/launch_remote.sh" "${SSH_HOST}" "${SSH_PORT}" | tee -a "${LOG}"
    ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p "${SSH_PORT}" root@"${SSH_HOST}"       "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits; ps -ef | grep run_worker_slice | grep -v grep; tail -n 40 /workspace/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_plan/logs/driver.log"       | tee -a "${LOG}"
    echo "[$(date +%F_%T)] launch_complete" | tee -a "${LOG}"
    exit 0
  fi

  echo "[$(date +%F_%T)] ssh_not_ready" | tee -a "${LOG}"
  sleep 20
done
