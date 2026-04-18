#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LEG_DIR}/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SELECT_LOG="${LOG_DIR}/wave2_select_${STAMP}.log"

cd "${LEG_DIR}"

while pgrep -af 'bash run_worker_slice.sh 0 0|bash run_worker_slice.sh 1 1' >/dev/null 2>&1; do
  sleep 10
done

BEST="$({
python3 - <<'PY'
import json
from pathlib import Path

best = None
for path in sorted((Path('evidence')).glob('worker[01]_*.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    for entry in payload.get('results', []):
        whale = entry.get('backends', {}).get('whale_fast', {})
        fa3 = entry.get('backends', {}).get('fa3', {})
        if not whale or not fa3:
            continue
        row = {
            'file': path.name,
            'whale_fb_ms': whale.get('fb_mean_ms'),
            'fa3_fb_ms': fa3.get('fb_mean_ms'),
            'env_bq': payload.get('env_bq'),
            'env_bq_maxnreg': payload.get('env_bq_maxnreg'),
        }
        key = (row['whale_fb_ms'], row['whale_fb_ms'] - row['fa3_fb_ms'])
        if best is None or key < (best['whale_fb_ms'], best['whale_fb_ms'] - best['fa3_fb_ms']):
            best = row
if best is None:
    raise SystemExit('no worker[01] evidence found')
print(best['env_bq'])
print(best['env_bq_maxnreg'])
print(best['file'])
print(best['whale_fb_ms'])
print(best['fa3_fb_ms'])
PY
} )"

BEST_DQ_TILE="$(printf '%s\n' "${BEST}" | sed -n '1p')"
BEST_DQ_MR="$(printf '%s\n' "${BEST}" | sed -n '2p')"
BEST_FILE="$(printf '%s\n' "${BEST}" | sed -n '3p')"
BEST_WHALE="$(printf '%s\n' "${BEST}" | sed -n '4p')"
BEST_FA3="$(printf '%s\n' "${BEST}" | sed -n '5p')"

{
  echo "selected_file=${BEST_FILE}"
  echo "selected_dq_tile=${BEST_DQ_TILE}"
  echo "selected_dq_maxnreg=${BEST_DQ_MR}"
  echo "selected_whale_fb_ms=${BEST_WHALE}"
  echo "selected_fa3_fb_ms=${BEST_FA3}"
} | tee "${SELECT_LOG}"

nohup bash "${LEG_DIR}/run_single_case.sh" \
  "wave2_validate_dq_${BEST_DQ_TILE}_dqmr_${BEST_DQ_MR}_dkdvmr_256" \
  0 10 200 "128,128,8,3" 256 "${BEST_DQ_TILE}" "${BEST_DQ_MR}" \
  > "${LOG_DIR}/wave2_validate_${STAMP}.log" 2>&1 &
echo "validate_pid=$!" | tee -a "${SELECT_LOG}"

nohup bash -lc "
  set -euo pipefail
  cd '${LEG_DIR}'
  for DKDV_MR in 160 192 224 256; do
    bash '${LEG_DIR}/run_single_case.sh' \
      \"wave2_dkdvmr_\${DKDV_MR}_dq_${BEST_DQ_TILE}_dqmr_${BEST_DQ_MR}\" \
      1 5 100 '128,128,8,3' \"\${DKDV_MR}\" '${BEST_DQ_TILE}' '${BEST_DQ_MR}'
  done
" > "${LOG_DIR}/wave2_dkdv_${STAMP}.log" 2>&1 &
echo "dkdv_followup_pid=$!" | tee -a "${SELECT_LOG}"
