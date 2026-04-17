#!/usr/bin/env bash
# whale_dkdv_warpspec — H7 driver.
# Phase 1: numerics check (warpspec ON vs SDPA reference) at small shapes.
# Phase 2: headline bench at 4 shapes, paired ON/OFF for the new knob.

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
echo "== ${TS}  H7 dkdv warp_specialize lever" | tee "${LOG}"
echo "== WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT}  WHALE_BWD_KV_WARPSPEC=${WHALE_BWD_KV_WARPSPEC}" | tee -a "${LOG}"

# -------------------------------------------------------------------------
# Phase 1 — numerics. whale_attn_fast (with WARPSPEC=on) vs SDPA reference.
# Reuses the harness from legs/2026-04-16_whale_fast_kernels/bench_numerics.py.
# -------------------------------------------------------------------------
echo "== Phase 1: numerics (whale_attn_fast vs SDPA, WARPSPEC=${WHALE_BWD_KV_WARPSPEC})" | tee -a "${LOG}"
rm -rf /root/.triton/cache 2>/dev/null || true
WHALE_BWD_KV_WARPSPEC=1 WHALE_BWD_VARIANT=baseline ${PY} \
    "${REPO_ROOT}/legs/2026-04-16_whale_fast_kernels/bench_numerics.py" \
    --kernel "vault.whale_kernel_triton:whale_attn_fast" \
    --label  "warpspec_on" \
    --out    "${LEG_DIR}/evidence/numerics_warpspec_on_${TS}.json" 2>&1 | tee -a "${LOG}"

# Control: same harness with knob OFF, to prove no regression.
WHALE_BWD_KV_WARPSPEC=0 WHALE_BWD_VARIANT=baseline ${PY} \
    "${REPO_ROOT}/legs/2026-04-16_whale_fast_kernels/bench_numerics.py" \
    --kernel "vault.whale_kernel_triton:whale_attn_fast" \
    --label  "warpspec_off" \
    --out    "${LEG_DIR}/evidence/numerics_warpspec_off_${TS}.json" 2>&1 | tee -a "${LOG}"

# -------------------------------------------------------------------------
# Phase 2 — headline bench. 4 shapes × {ON, OFF} × {whale_fast, fa3, sdpa}.
# Shapes: 4,2048,8,4,64 ; 4,4096,8,4,64 ; 2,8192,8,4,64 ; 4,2048,16,16,64.
# -------------------------------------------------------------------------
SHAPES="4,2048,8,4,64;4,4096,8,4,64;2,8192,8,4,64;4,2048,16,16,64"

for KNOB in 1 0; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    LABEL="warpspec_${KNOB}"
    echo "== Phase 2: bench ${LABEL}" | tee -a "${LOG}"
    WHALE_BWD_KV_WARPSPEC=${KNOB} WHALE_BWD_VARIANT=baseline ${PY} \
        "${LEG_DIR}/bench_stable.py" \
        --shapes   "${SHAPES}" \
        --backends "whale_fast,fa3,sdpa" \
        --rounds   "${WHALE_BENCH_ROUNDS}" \
        --iters    "${WHALE_BENCH_ITERS}" \
        --label    "${LABEL}" \
        --out      "${LEG_DIR}/evidence/bench_${LABEL}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

echo "== done. log: ${LOG}"
