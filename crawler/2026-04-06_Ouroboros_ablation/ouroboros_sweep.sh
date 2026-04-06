#!/usr/bin/env bash
set -euo pipefail

# Ouroboros 3-arm ablation sweep + control
# All seed=300, 4xGPU, 600s wallclock
# Base: Ouroboros PR #1308 config (9F+1Cx2, loop-aware GPTQ, QK4, brotli)
#
# Control: exact Ouroboros baseline
# Arm A: Noisy QAT (int6-calibrated, crawler blocks only) — PR #363 technique
# Arm B: CRAWLER_QUANT_INT8=1 (int8 for crawler blocks, reduce quant compounding)
# Arm C: SCORE contractive dt=0.5 on crawler loops (arXiv 2603.10544)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "=== OUROBOROS ABLATION SWEEP START $(date) ==="

for arm in ouroboros_control ouroboros_arm_a_noisy_qat ouroboros_arm_b_crawler_int8 ouroboros_arm_c_contractive; do
    # Fix filename for arm B
    script="${SCRIPT_DIR}/${arm}.sh"
    if [ ! -f "${script}" ] && [ "${arm}" = "ouroboros_arm_b_crawler_int8" ]; then
        script="${SCRIPT_DIR}/ouroboros_arm_b_mixedbit.sh"
    fi
    echo ""
    echo "=== Running ${arm} $(date) ==="
    bash "${script}" || echo "WARNING: ${arm} failed with exit code $?"
    echo "=== Finished ${arm} $(date) ==="
done

echo ""
echo "=== OUROBOROS ABLATION SWEEP COMPLETE $(date) ==="
