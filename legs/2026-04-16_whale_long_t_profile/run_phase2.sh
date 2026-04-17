#!/usr/bin/env bash
# whale_long_t_profile — Phase 2.
# Kineto trace whale (with VARIANT from $1) vs FA3 at T=8192.

set -euo pipefail

LEG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LEG_DIR}/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
source "${LEG_DIR}/tracked_env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY=/venv/main/bin/python3

VARIANT="${1:-auto}"
SHAPE="${2:-2,8192,8,4,64}"
STEPS="${3:-30}"

LOG="${LEG_DIR}/logs/phase2_${TS}.log"
mkdir -p "${LEG_DIR}/logs" "${LEG_DIR}/evidence"
echo "== ${TS} Phase 2: kineto profile (variant=${VARIANT} shape=${SHAPE})" | tee "${LOG}"

rm -rf /root/.triton/cache 2>/dev/null || true

${PY} "${LEG_DIR}/profile_long_t.py" \
    --backend whale --variant "${VARIANT}" --shape "${SHAPE}" --steps "${STEPS}" \
    --out "${LEG_DIR}/evidence/profile_whale_${VARIANT}_${TS}.json" 2>&1 | tee -a "${LOG}"

${PY} "${LEG_DIR}/profile_long_t.py" \
    --backend fa3 --shape "${SHAPE}" --steps "${STEPS}" \
    --out "${LEG_DIR}/evidence/profile_fa3_${TS}.json" 2>&1 | tee -a "${LOG}"

echo "== Phase 2 done. log: ${LOG}"
