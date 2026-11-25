#!/usr/bin/env bash
# Dispatch the Midnight 12L clean campaign.
# Usage: bash scripts/run_midnight_12l_lane.sh gate|run [seed] [nproc_per_node]
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LEG="${REPO_ROOT}/legs/2026-04-12_midnight_12l_clean"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_midnight_12l_lane.sh gate [seed] [nproc_per_node]
  bash scripts/run_midnight_12l_lane.sh run  [seed] [nproc_per_node]

Runs ${LEG}/gate.sh or run.sh.
SEED and NPROC_PER_NODE are the only shell overrides. Training conditions
live in that leg's tracked_env.sh.
EOF
}

mode="${1:-}"
if [[ "${mode}" == "-h" || "${mode}" == "--help" || -z "${mode}" ]]; then
  usage
  [[ -n "${mode}" ]]
  exit $?
fi
if [[ "${mode}" != "gate" && "${mode}" != "run" ]]; then
  usage >&2
  echo "FATAL: mode must be gate or run, got: ${mode}" >&2
  exit 1
fi

if [[ $# -ge 2 ]]; then
  export SEED="$2"
fi
if [[ $# -ge 3 ]]; then
  export NPROC_PER_NODE="$3"
fi
if [[ $# -gt 3 ]]; then
  echo "FATAL: too many arguments" >&2
  usage >&2
  exit 1
fi

echo "lane: midnight_12l leg=${LEG} mode=${mode} seed=${SEED:-<script default>} nproc=${NPROC_PER_NODE:-<script default>}"
exec bash "${LEG}/${mode}.sh"
