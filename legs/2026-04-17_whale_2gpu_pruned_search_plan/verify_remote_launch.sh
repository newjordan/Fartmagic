#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${1:?ssh host required}"
SSH_PORT="${2:?ssh port required}"
SSH_KEY="/home/frosty40/.ssh/id_ed25519_apollo"
REMOTE_ROOT="/workspace/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_plan"

/usr/bin/ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p "${SSH_PORT}" root@"${SSH_HOST}"   "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits &&    ps -ef | grep run_worker_slice | grep -v grep &&    echo ---DRIVER--- && tail -n 80 ${REMOTE_ROOT}/logs/driver.log &&    echo ---WORKERS--- && ls -1 ${REMOTE_ROOT}/logs/driver_worker*.log 2>/dev/null || true &&    echo ---EVIDENCE--- && find ${REMOTE_ROOT}/evidence -maxdepth 1 -type f | wc -l"
