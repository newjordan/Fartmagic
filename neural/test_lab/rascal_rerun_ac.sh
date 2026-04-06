#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "=== RASCAL RERUN A+C $(date) ==="
echo "=== Arm A: trigram ==="
bash "${SCRIPT_DIR}/arm_a_trigram_v2.sh" || echo "WARNING: arm_a failed $?"
echo "=== Arm C: gated attention ==="
bash "${SCRIPT_DIR}/arm_c_gated_attn_v2.sh" || echo "WARNING: arm_c failed $?"
echo "=== DONE $(date) ==="
