#!/usr/bin/env bash
# whale_dkdv_persistent — H6 benchmark driver.
# Numerics check on 4 primary shapes, then head-to-head
# baseline(dkdv) vs baseline+persistent(dkdv) vs FA3 vs SDPA.
#
# IMPORTANT: this run assumes the vault_patch.md edits have been applied
# to vault/whale_kernel_triton.py. If they have not, the
# WHALE_BWD_KV_PERSISTENT env var is a no-op and the persistent results
# will equal the baseline results (safe, but not a valid H6 test).

set -euo pipefail

LEG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LEG_DIR}/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
source "${LEG_DIR}/tracked_env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY=/venv/main/bin/python3

LOG="${LEG_DIR}/logs/run_${TS}.log"
mkdir -p "${LEG_DIR}/logs" "${LEG_DIR}/evidence"
echo "== ${TS}  WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT}  WHALE_BWD_KV_PERSISTENT=${WHALE_BWD_KV_PERSISTENT}" | tee "${LOG}"

# Reuse the sibling leg's bench_stable.py — same harness, same backends.
BENCH="${REPO_ROOT}/legs/2026-04-16_whale_bwd_tma_dkdv/bench_stable.py"
NUMERICS="${REPO_ROOT}/legs/2026-04-16_whale_fast_kernels/bench_numerics.py"

# Primary 4 shapes.
SHAPES="2,8192,8,4,64;4,4096,8,4,64;4,2048,8,4,64;4,2048,16,16,64"

# 1. Numerics check with persistent enabled.
rm -rf /root/.triton/cache 2>/dev/null || true
WHALE_BWD_VARIANT=baseline WHALE_BWD_KV_PERSISTENT=1 ${PY} "${NUMERICS}" \
    --kernel vault.whale_kernel_triton:whale_attn_fast \
    --label persistent_numerics \
    --out "${LEG_DIR}/evidence/persistent_numerics_${TS}.json" 2>&1 | tee -a "${LOG}"

# 2. Head-to-head on 4 primary shapes: baseline vs baseline+persistent.
for PERSIST in 0 1; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    LABEL="sweep_baseline_persist${PERSIST}"
    WHALE_BWD_VARIANT=baseline WHALE_BWD_KV_PERSISTENT=${PERSIST} ${PY} "${BENCH}" \
        --shapes "${SHAPES}" \
        --backends "whale_fast,fa3,sdpa" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label  "${LABEL}" \
        --out    "${LEG_DIR}/evidence/${LABEL}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

# 3. Compare to the existing auto/fused_delta winner (shared baseline).
rm -rf /root/.triton/cache 2>/dev/null || true
WHALE_BWD_VARIANT=auto WHALE_BWD_KV_PERSISTENT=0 ${PY} "${BENCH}" \
    --shapes "${SHAPES}" \
    --backends "whale_fast,fa3,sdpa" \
    --rounds "${WHALE_BENCH_ROUNDS}" \
    --iters  "${WHALE_BENCH_ITERS}" \
    --label  "sweep_auto_reference" \
    --out    "${LEG_DIR}/evidence/sweep_auto_reference_${TS}.json" 2>&1 | tee -a "${LOG}"

echo "== done. log: ${LOG}"
