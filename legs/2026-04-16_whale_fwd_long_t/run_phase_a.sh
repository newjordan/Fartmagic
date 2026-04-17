#!/usr/bin/env bash
# whale_fwd_long_t — Lever C Phase A driver (PATCH-FREE).
#
# Goal: autotune-sweep the existing TMA fwd kernel at long T and compare
# whale_fast (WHALE_FWD_VARIANT=tma) vs FA3 at three shapes.
#
# Phase A is patch-free. No vault edit, no forced config. The TMA forward
# autotune config grid (`_fwd_tma_configs`, vault/whale_kernel_triton.py:81-94)
# covers BM,BN in {64,128} x W in {4,8} x S in {2,3,4} = 24 configs; the
# triton cache is wiped between shapes so each shape picks its own winner
# fresh under the long-T key `(D, IS_CAUSAL, NUM_HEADS, NUM_KV_HEADS, T_MAX)`.
#
# Shapes (per Lever C brief):
#   (B,T,H,KV,D) = (2, 4096, 8, 4, 64)
#   (B,T,H,KV,D) = (2, 8192, 8, 4, 64)
#   (B,T,H,KV,D) = (1,16384, 8, 4, 64)
#
# Bench harness: legs/2026-04-16_whale_fwd_long_t/bench_stable.py
# Backends: whale_fast (uses WHALE_FWD_VARIANT=tma from tracked_env.sh), fa3.
# Evidence:  legs/2026-04-16_whale_fwd_long_t/evidence/phase_a_<shape>_<TS>.json

set -euo pipefail

LEG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LEG_DIR}/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
source "${LEG_DIR}/tracked_env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY=/venv/main/bin/python3

# Lever C Phase A: TMA forward, no forced tile. Unset any stale force so the
# autotuner is free to pick the long-T winner per shape.
export WHALE_FWD_VARIANT=tma
unset WHALE_FWD_TMA_CONFIG   || true
unset WHALE_FWD_TMA_WARPSPEC || true

LOG="${LEG_DIR}/logs/phase_a_${TS}.log"
mkdir -p "${LEG_DIR}/logs" "${LEG_DIR}/evidence"
echo "== ${TS}  Lever C Phase A — whale_fast(tma fwd) vs FA3 at long T" | tee "${LOG}"
echo "   WHALE_FWD_VARIANT=${WHALE_FWD_VARIANT}" | tee -a "${LOG}"
echo "   rounds=${WHALE_BENCH_ROUNDS}  iters=${WHALE_BENCH_ITERS}" | tee -a "${LOG}"

# One shape per call so triton-cache wipe forces fresh autotune per shape.
# Tag format for evidence: phase_a_B_T_H_KV_D.
SHAPES=(
    "2,4096,8,4,64"
    "2,8192,8,4,64"
    "1,16384,8,4,64"
)

for SHAPE in "${SHAPES[@]}"; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    SAFE_TAG="$(echo "${SHAPE}" | tr ',' '_')"
    OUT_JSON="${LEG_DIR}/evidence/phase_a_${SAFE_TAG}_${TS}.json"
    echo "" | tee -a "${LOG}"
    echo "---- shape=${SHAPE}  out=${OUT_JSON}" | tee -a "${LOG}"
    ${PY} "${LEG_DIR}/bench_stable.py" \
        --shape "${SHAPE}" \
        --backends "whale_fast,fa3" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label  "phase_a_${SAFE_TAG}" \
        --out    "${OUT_JSON}" 2>&1 | tee -a "${LOG}"
done

echo "" | tee -a "${LOG}"
echo "== Lever C Phase A done. log: ${LOG}"
echo "== evidence JSONs: ${LEG_DIR}/evidence/phase_a_*_${TS}.json"
