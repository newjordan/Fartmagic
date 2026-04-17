#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/tracked_env.sh"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/evidence"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${SCRIPT_DIR}/evidence/combined_validation_${TS}.json"
LOG="${SCRIPT_DIR}/logs/combined_validation_${TS}.log"
BENCH="${REPO_ROOT}/legs/2026-04-16_whale_bwd_ablations/bench_stable.py"

{
  echo "[${TS}] combined validation start"
  echo "bench=${BENCH}"
  echo "shape=${BENCH_SHAPE} backends=${BENCH_BACKENDS} rounds=${BENCH_ROUNDS} iters=${BENCH_ITERS}"
  echo "dkdv=${DKDV_WINNER_CONFIG}"
  echo "dq_tile=${DQ_WINNER_CONFIG}"
  echo "dq_maxnreg=${DQ_WINNER_MAXNREG}"
  /venv/main/bin/python3 "${BENCH}" \
    --label "whale_combined_validation_t2048" \
    --shape "${BENCH_SHAPE}" \
    --backends "${BENCH_BACKENDS}" \
    --rounds "${BENCH_ROUNDS}" \
    --iters "${BENCH_ITERS}" \
    --out "${OUT}"
  echo "[${TS}] combined validation done"
} >"${LOG}" 2>&1

printf '%s\n' "${OUT}"
