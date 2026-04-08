#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/sidelane/scripts/run_v1_4_submission_sweep_brotli.sh"

if [[ ! -x "${RUNNER}" ]]; then
  echo "Missing executable runner: ${RUNNER}" >&2
  echo "Re-pull TEST_LAB and confirm test_lab assets are present." >&2
  exit 1
fi

export PROJECT_CODENAME="${PROJECT_CODENAME:-longworm}"
export SUBMISSION_PROFILE="${SUBMISSION_PROFILE:-track_10min_16mb}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export ARM_ONLY="${ARM_ONLY:-36_v1_6_longworm_fast_l8_d480_h12_kv4_non_ngram_brotli}"
export TRACK_4K_BPB="${TRACK_4K_BPB:-0}"
export LEADERBOARD_METRIC_PREF="${LEADERBOARD_METRIC_PREF:-submission}"

if [[ -z "${TRAIN_PY:-}" ]]; then
  export TRAIN_PY="${SCRIPT_DIR}/../../experiments/Longworm/train_longworm.py"
fi

echo "============================================================"
echo "Longworm fast L8 submission launcher"
echo "train_py: ${TRAIN_PY}"
echo "arm_only: ${ARM_ONLY}"
echo "submission_profile: ${SUBMISSION_PROFILE}"
echo "track_4k_bpb: ${TRACK_4K_BPB}"
echo "leaderboard_metric_pref: ${LEADERBOARD_METRIC_PREF}"
echo "============================================================"

exec bash "${RUNNER}" "$@"
