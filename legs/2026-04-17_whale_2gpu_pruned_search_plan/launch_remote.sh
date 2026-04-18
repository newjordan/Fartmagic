#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${1:?ssh host required}"
SSH_PORT="${2:?ssh port required}"
REMOTE_ROOT="/workspace/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_plan"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_KEY="/home/frosty40/.ssh/id_ed25519_apollo"

ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p "${SSH_PORT}" root@"${SSH_HOST}" "mkdir -p ${REMOTE_ROOT}/{logs,evidence,runtime,frozen_vault}"
scp -i "${SSH_KEY}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -P "${SSH_PORT}" -r \
  "${LOCAL_DIR}/PLAN.md" \
  "${LOCAL_DIR}/hypothesis.md" \
  "${LOCAL_DIR}/tracked_env.sh" \
  "${LOCAL_DIR}/RESULTS.md" \
  "${LOCAL_DIR}/prepare_remote_env.sh" \
  "${LOCAL_DIR}/run_worker_slice.sh" \
  "${LOCAL_DIR}/run_on_pod.sh" \
  "${LOCAL_DIR}/bench_stable_scratch.py" \
  "${LOCAL_DIR}/summarize_results.py" \
  "${LOCAL_DIR}/validate_candidate_remote.sh" \
  "${LOCAL_DIR}/verify_remote_launch.sh" \
  "${LOCAL_DIR}/frozen_vault" \
  root@"${SSH_HOST}":"${REMOTE_ROOT}/"
ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p "${SSH_PORT}" root@"${SSH_HOST}" \
  "cd ${REMOTE_ROOT} && chmod +x tracked_env.sh prepare_remote_env.sh run_worker_slice.sh run_on_pod.sh validate_candidate_remote.sh verify_remote_launch.sh && bash prepare_remote_env.sh && nohup bash run_on_pod.sh > logs/driver.log 2>&1 & echo PID=\$!"
