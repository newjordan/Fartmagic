#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "=== MEGARASCAL SPEED ABLATION SWEEP ==="
echo "4 arms × 2000 steps × 1 GPU — measuring ms/step"
echo ""

for script in \
    "${SCRIPT_DIR}/megarascal_control.sh" \
    "${SCRIPT_DIR}/megarascal_arm_a_cudagraphs.sh" \
    "${SCRIPT_DIR}/megarascal_arm_b_triton_act.sh" \
    "${SCRIPT_DIR}/megarascal_arm_c_fused_mlp.sh"; do
    echo "========================================"
    echo "Running: $(basename "${script}")"
    echo "========================================"
    bash "${script}" || echo "FAILED: $(basename "${script}")"
    echo ""
done

echo "=== SWEEP COMPLETE ==="
