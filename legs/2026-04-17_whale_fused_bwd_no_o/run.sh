#!/usr/bin/env bash
# whale_fused_bwd_no_o — A/B bench driver.
# Runs (1) numerics check vs SDPA, (2) headline head-to-head over 4
# primary shapes: fused_delta vs fused_bwd (H5) vs fused_bwd_no_o (this
# leg) vs FA3 vs SDPA.
#
# PRECONDITION: vault_patch.md edits must be applied to
# vault/whale_kernel_triton.py before this script is launched. The
# preflight check below verifies the new kernel symbol is importable.
#
# Reuses bench_stable.py and bench_numerics.py from sibling legs — this
# leg adds no new benchmark code, just a new variant to the sweep.

set -euo pipefail

LEG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LEG_DIR}/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
source "${LEG_DIR}/tracked_env.sh"

LOG="${LEG_DIR}/logs/run_${TS}.log"
mkdir -p "${LEG_DIR}/logs" "${LEG_DIR}/evidence"
echo "== ${TS}  whale_fused_bwd_no_o" | tee "${LOG}"

# Startup safety preflight (see CLAUDE.md §Startup Safety Protocol).
pwd                                         | tee -a "${LOG}"
git remote -v                               | tee -a "${LOG}"
git rev-parse --abbrev-ref HEAD             | tee -a "${LOG}"
test -f "${LEG_DIR}/run.sh"                 | tee -a "${LOG}"
test -f vault/whale_kernel_triton.py        | tee -a "${LOG}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY=/venv/main/bin/python3

# Patch-applied preflight: require that the new kernel symbol exists.
${PY} - <<'PYEOF' 2>&1 | tee -a "${LOG}"
import sys
import importlib
mod = importlib.import_module("vault.whale_kernel_triton")
if not hasattr(mod, "_attn_bwd_fused_no_o_kernel"):
    print("PREFLIGHT FAIL: _attn_bwd_fused_no_o_kernel not in vault/whale_kernel_triton.py")
    print("Apply vault_patch.md first.")
    sys.exit(2)
print("PREFLIGHT OK: _attn_bwd_fused_no_o_kernel present.")
PYEOF

# Fresh autotune cache so the new config list gets a clean pick.
rm -rf /root/.triton/cache 2>/dev/null || true

# Primary 4 shapes for this leg.
PRIMARY_SHAPES="4,2048,8,4,64;4,4096,8,4,64;2,8192,8,4,64;2,2048,8,4,128"

# 1. Numerics check vs SDPA across primary shapes.
WHALE_BWD_VARIANT=fused_bwd_no_o ${PY} \
    legs/2026-04-16_whale_fast_kernels/bench_numerics.py \
    --kernel vault.whale_kernel_triton:whale_attn_fast \
    --label fused_bwd_no_o_numerics \
    --out "${LEG_DIR}/evidence/fused_bwd_no_o_numerics_${TS}.json" \
    2>&1 | tee -a "${LOG}"

# 2. Head-to-head sweep across the 4 primary shapes. One pass per variant
#    so each gets its own warm autotune cache. Order: delta first
#    (known winner) to establish FA3 baseline, then H5 fused_bwd, then
#    the new fused_bwd_no_o.
for VARIANT in fused_delta fused_bwd fused_bwd_no_o; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    WHALE_BWD_VARIANT=${VARIANT} ${PY} \
        legs/2026-04-16_whale_bwd_persistent_atomic/bench_stable.py \
        --shapes "${PRIMARY_SHAPES}" \
        --backends "whale_fast,fa3,sdpa" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label  "sweep_${VARIANT}" \
        --out    "${LEG_DIR}/evidence/sweep_${VARIANT}_${TS}.json" \
        2>&1 | tee -a "${LOG}"
done

echo "== done. log: ${LOG}"
