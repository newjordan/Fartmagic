#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SEEDS="${SEEDS:-4 300 444}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-tone_comp}"

for seed in ${SEEDS}; do
  run_id="${RUN_ID_PREFIX}_s${seed}"
  echo "[TON-E] running seed=${seed} run_id=${run_id}"
  SEED="${seed}" RUN_ID="${run_id}" bash "${SCRIPT_DIR}/run_competition.sh"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[TON-E] dry-run complete (skipped submission.json build)"
  exit 0
fi

python3 "${SCRIPT_DIR}/build_submission_json.py" \
  --log-glob "logs/${RUN_ID_PREFIX}_s*.txt" \
  --output "${SCRIPT_DIR}/submission.json" \
  --name "${SUBMISSION_NAME:-TON-E Rhythm Crawler}" \
  --author "${SUBMISSION_AUTHOR:-Frosty40}" \
  --github-id "${SUBMISSION_GITHUB_ID:-newjordan}" \
  --hardware "${SUBMISSION_HARDWARE:-8xH100 SXM}"

echo "[TON-E] wrote ${SCRIPT_DIR}/submission.json"
