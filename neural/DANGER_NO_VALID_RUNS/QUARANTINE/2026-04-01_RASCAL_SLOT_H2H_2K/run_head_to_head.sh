#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/neural/2026-04-01_RASCAL_SLOT_H2H_2K"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"

cd "${REPO_ROOT}"

echo "=== HEAD TO HEAD: RASCAL II 2K vs SLOT 2K/H2H ==="
echo "seed=${SEED} nproc=${NPROC}"
echo

echo "--- RUN 1/2: RASCAL II 2K ---"
SEED="${SEED}" NPROC_PER_NODE="${NPROC}" bash "${SCRIPT_DIR}/run_rascal_ii_2k.sh"

echo
echo "--- RUN 2/2: SLOT 2K/H2H ---"
SEED="${SEED}" NPROC_PER_NODE="${NPROC}" bash "${SCRIPT_DIR}/run.sh"

BASE_LOG="$(ls -1t "${SCRIPT_DIR}/logs"/rascal_ii_2k_seed${SEED}_*.log 2>/dev/null | head -n1 || true)"
SLOT_LOG="$(ls -1t "${SCRIPT_DIR}/logs"/slot_h2h_2k_seed${SEED}_*.log 2>/dev/null | head -n1 || true)"

echo
echo "=== SUMMARY ==="
if [[ -n "${BASE_LOG}" ]]; then
  echo "--- RASCAL II (${BASE_LOG}) ---"
  grep -E "step:500/|step:2000/|Serialized model|Code size|Serialized model int6\\+|Total submission size int6\\+|final_sliding_window.*_exact" "${BASE_LOG}" | tail -n 20 || true
fi
if [[ -n "${SLOT_LOG}" ]]; then
  echo "--- SLOT H2H (${SLOT_LOG}) ---"
  grep -E "step:500/|step:2000/|Serialized model|Code size|Serialized model int6\\+|Total submission size int6\\+|h2h_" "${SLOT_LOG}" | tail -n 30 || true
fi
