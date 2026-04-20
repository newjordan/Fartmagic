#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${LEG_DIR}/state"
LOG_DIR="${LEG_DIR}/logs"
EVID_DIR="${LEG_DIR}/evidence"
TASKS_FILE="${STATE_DIR}/tasks.txt"
DONE_FILE="${STATE_DIR}/done.txt"
GPU_ID="${1:?gpu required}"
WORKER_LOG="${LOG_DIR}/worker_${GPU_ID}.log"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${EVID_DIR}"
source "${LEG_DIR}/tracked_env.sh"

log() {
  echo "[$(date +%F_%T)] $*" | tee -a "${WORKER_LOG}"
}

pop_task() {
  /venv/main/bin/python3 - "${TASKS_FILE}" <<'PY'
import fcntl
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.touch(exist_ok=True)
with path.open('r+', encoding='utf-8') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    lines = [line.rstrip('\n') for line in f if line.strip()]
    task = lines[0] if lines else ''
    rest = lines[1:] if lines else []
    f.seek(0)
    f.truncate(0)
    if rest:
        f.write('\n'.join(rest) + '\n')
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
print(task)
PY
}

prepend_task() {
  local task="$1"
  /venv/main/bin/python3 - "${TASKS_FILE}" "${DONE_FILE}" "${task}" <<'PY'
import fcntl
import sys
from pathlib import Path
queue = Path(sys.argv[1])
done = Path(sys.argv[2])
task = sys.argv[3]
queue.touch(exist_ok=True)
done.touch(exist_ok=True)
with queue.open('r+', encoding='utf-8') as qf:
    fcntl.flock(qf.fileno(), fcntl.LOCK_EX)
    queued = [line.rstrip('\n') for line in qf if line.strip()]
    done_tasks = {line.strip() for line in done.read_text(encoding='utf-8').splitlines() if line.strip()}
    if task not in queued and task not in done_tasks:
        queued = [task] + queued
    qf.seek(0)
    qf.truncate(0)
    if queued:
        qf.write('\n'.join(queued) + '\n')
    fcntl.flock(qf.fileno(), fcntl.LOCK_UN)
PY
}

mark_done() {
  printf '%s\n' "$1" >> "${DONE_FILE}"
}

latest_json() {
  local pattern="$1"
  ls -1t ${EVID_DIR}/${pattern} 2>/dev/null | head -n 1
}

probe_is_green() {
  local path="$1"
  /venv/main/bin/python3 - "${path}" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
payload = json.loads(p.read_text())
entry = payload['results'][0]
whale = entry['backends']['whale_fast']['fb_mean_ms']
fa3 = entry['backends']['fa3']['fb_mean_ms']
raise SystemExit(0 if whale < fa3 else 1)
PY
}

summarize_json() {
  local path="$1"
  /venv/main/bin/python3 - "${path}" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
payload = json.loads(p.read_text())
entry = payload['results'][0]
whale = entry['backends']['whale_fast']['fb_mean_ms']
fa3 = entry['backends']['fa3']['fb_mean_ms']
print(f"whale_fb={whale:.12f} fa3_fb={fa3:.12f} delta_ms={whale-fa3:+.12f}")
PY
}

run_task() {
  local task="$1"
  local family phase dq_tile dkdv_tile dkdv_mr qmr
  IFS='|' read -r family phase dq_tile dkdv_tile dkdv_mr qmr <<< "${task}"
  log "gpu=${GPU_ID} start ${task}"
  bash "${LEG_DIR}/run_cross_gpu_case.sh" "${family}" "${phase}" "${dq_tile}" "${dkdv_tile}" "${dkdv_mr}" "${qmr}" "${GPU_ID}" >> "${WORKER_LOG}" 2>&1
  local pattern
  pattern="cross_${family}_${phase}_dq_${dq_tile//,/_}_dkdv_${dkdv_tile//,/_}_mr_${dkdv_mr}_qmr_${qmr}_*.json"
  local path
  path="$(latest_json "${pattern}")"
  if [[ -n "${path}" ]]; then
    log "gpu=${GPU_ID} done ${task} $(summarize_json "${path}")"
    if [[ "${phase}" == "probe" ]] && probe_is_green "${path}"; then
      prepend_task "${family}|validate|${dq_tile}|${dkdv_tile}|${dkdv_mr}|${qmr}"
      log "gpu=${GPU_ID} promoted ${family}|validate|${dq_tile}|${dkdv_tile}|${dkdv_mr}|${qmr}"
    fi
  else
    log "gpu=${GPU_ID} missing_json ${task}"
  fi
  mark_done "${task}"
}

while true; do
  task="$(pop_task)"
  [[ -z "${task}" ]] && break
  run_task "${task}"
done

log "gpu=${GPU_ID} worker_done"
