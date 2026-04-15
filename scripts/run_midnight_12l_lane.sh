#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_midnight_12l_lane.sh <gate|run|full> [seed] [nproc_per_node]

Examples:
  bash scripts/run_midnight_12l_lane.sh gate
  bash scripts/run_midnight_12l_lane.sh gate 444 1
  bash scripts/run_midnight_12l_lane.sh run 444 8
USAGE
}

MODE="${1:-gate}"
SEED_ARG="${2:-}"
NPROC_ARG="${3:-}"

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LEG_REL="legs/2026-04-14_12l_eval_fix"
LEG_DIR="${REPO_ROOT}/${LEG_REL}"

[[ -d "${LEG_DIR}" ]] || { echo "FATAL: missing leg directory: ${LEG_REL}" >&2; exit 1; }

case "${MODE}" in
  gate)
    TARGET="${LEG_DIR}/gate.sh"
    ;;
  run|full)
    TARGET="${LEG_DIR}/run.sh"
    ;;
  *)
    echo "FATAL: mode must be one of gate|run|full" >&2
    usage >&2
    exit 1
    ;;
esac

[[ -x "${TARGET}" ]] || { echo "FATAL: runner script is not executable: ${TARGET}" >&2; exit 1; }

# Only operational selectors are allowed here. Experiment conditions live in tracked leg files.
if [[ -n "${SEED_ARG}" ]]; then
  export SEED="${SEED_ARG}"
fi
if [[ -n "${NPROC_ARG}" ]]; then
  export NPROC_PER_NODE="${NPROC_ARG}"
fi

echo "LEG=${LEG_REL}"
echo "MODE=${MODE}"
echo "SEED=${SEED:-default}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE:-default}"

cd "${REPO_ROOT}"
bash "${TARGET}"
