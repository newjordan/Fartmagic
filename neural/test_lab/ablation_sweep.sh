#!/usr/bin/env bash
set -euo pipefail

# 5-arm ablation sweep: all vs mixed-int baseline (attn=5, mlp=6, embed=8)
# Each arm changes ONE variable. All seed=300, 4xGPU, 600s wallclock.
#
# Control already run: today.sh (mixed-int baseline)
# Arm A: TRIGRAM=1 (env var, base train_gpt.py)
# Arm B: mu-centering (code change: subtract mean output embed post-step)
# Arm C: GATED_ATTENTION=1 (env var, base train_gpt.py)
# Arm D: HEQ quantile scale selection (code change: export only)
# Arm E: DDL residual (code change: rank-1 delta operator on residuals)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ABLATION SWEEP START $(date) ==="

for arm in arm_a_trigram arm_b_mu_centering arm_c_gated_attn arm_d_heq arm_e_ddl; do
    echo ""
    echo "=== Running ${arm} $(date) ==="
    bash "${SCRIPT_DIR}/${arm}.sh" || echo "WARNING: ${arm} failed with exit code $?"
    echo "=== Finished ${arm} $(date) ==="
done

echo ""
echo "=== ABLATION SWEEP COMPLETE $(date) ==="
