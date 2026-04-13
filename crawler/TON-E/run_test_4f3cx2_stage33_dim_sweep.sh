#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_test_4f3cx2_stage33_d480.sh"

if [[ ! -f "${BASE_RUNNER}" ]]; then
  echo "[TON-E] missing base runner: ${BASE_RUNNER}" >&2
  exit 1
fi

SEED="${SEED:-4}"
WORLD_SIZE="${WORLD_SIZE:-1}"
DIMS="${DIMS:-512 448 384}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-tone_test_4f3cx2_stage33}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

echo "[TON-E] dim sweep start"
echo "[TON-E] seed=${SEED} world_size=${WORLD_SIZE} dims=${DIMS}"

for dim in ${DIMS}; do
  run_id="${RUN_ID_PREFIX}_d${dim}_s${SEED}"
  echo "[TON-E] running dim=${dim} run_id=${run_id}"
  if ! SEED="${SEED}" WORLD_SIZE="${WORLD_SIZE}" MODEL_DIM="${dim}" RUN_ID="${run_id}" bash "${BASE_RUNNER}"; then
    echo "[TON-E] failed dim=${dim} run_id=${run_id}" >&2
    if [[ "${STOP_ON_FAIL}" == "1" ]]; then
      exit 1
    fi
  fi
done

echo "[TON-E] dim sweep complete"
