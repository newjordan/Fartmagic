#!/usr/bin/env bash
# Phases A and B are patch-free; Phase C requires vault patch in vault_patch.md
# whale_fwd_long_t — H9 driver.
#
# Phase A (cheap, no kernel change):
#   bench WHALE_FWD_VARIANT={default, tma} at T in {2048, 4096, 8192}.
#
# Phase B (cheap, no kernel change):
#   bench WHALE_FWD_VARIANT=tma with FA3-like forced configs at T=8192.
#
# Phase C (gated by vault patch, doc-only here):
#   bench WHALE_FWD_TMA_WARPSPEC=1 at T=8192. Skipped automatically when
#   the vault has not been patched (the env is honored only by patched code).

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
echo "== ${TS}  whale_fwd_long_t — H9 driver" | tee "${LOG}"

SHAPES_A="4,2048,8,4,64;4,4096,8,4,64;2,8192,8,4,64"
SHAPE_LONG="2,8192,8,4,64"

# ---------------------------------------------------------------------------
# Phase A: variant sweep, no forced config. Compares default vs tma autotune
# winners at three T values to localize where TMA starts to pay off.
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "## Phase A — variant sweep" | tee -a "${LOG}"
for VARIANT in default tma; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    WHALE_FWD_VARIANT=${VARIANT} ${PY} "${LEG_DIR}/bench_stable.py" \
        --shapes "${SHAPES_A}" \
        --backends "whale,fa3,sdpa" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label "phaseA_${VARIANT}" \
        --out "${LEG_DIR}/evidence/phaseA_${VARIANT}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

# ---------------------------------------------------------------------------
# Phase B: TMA variant with FA3-like forced tiles at T=8192.
# WHALE_FWD_TMA_CONFIG = "BM,BN,W,S" routes through _env_force("FWD_TMA")
# (vault/whale_kernel_triton.py:85). Each spin must wipe the triton cache
# so the autotune is forced to the single config under test.
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "## Phase B — TMA forced-config sweep at T=8192" | tee -a "${LOG}"
for CFG in ${WHALE_FWD_TMA_PHASEB_CONFIGS}; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    SAFE_TAG="$(echo "${CFG}" | tr ',' '_')"
    WHALE_FWD_VARIANT=tma WHALE_FWD_TMA_CONFIG="${CFG}" \
        ${PY} "${LEG_DIR}/bench_stable.py" \
        --shapes "${SHAPE_LONG}" \
        --backends "whale,fa3" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label "phaseB_tma_${SAFE_TAG}" \
        --out "${LEG_DIR}/evidence/phaseB_tma_${SAFE_TAG}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

# ---------------------------------------------------------------------------
# Phase C: TMA + warp_specialize on the K/V loop (H9b).
# ONLY meaningful if vault_patch.md has been applied to vault/. If the env
# knob is honored, the patched kernel exposes a constexpr WARPSPEC=True
# branch; if the vault is unpatched, the env is inert and the run is a no-op
# vs Phase A's tma baseline (still useful: confirms inertness).
# Each forced config tested with warpspec on, since warp_specialize=True
# requires num_stages>=2 and num_warps in {4,8} on Triton 3.6 cu130.
# ---------------------------------------------------------------------------
echo "" | tee -a "${LOG}"
echo "## Phase C — TMA + warp_specialize at T=8192 (gated by vault patch)" | tee -a "${LOG}"
for CFG in ${WHALE_FWD_TMA_PHASEB_CONFIGS}; do
    rm -rf /root/.triton/cache 2>/dev/null || true
    SAFE_TAG="$(echo "${CFG}" | tr ',' '_')"
    WHALE_FWD_VARIANT=tma WHALE_FWD_TMA_CONFIG="${CFG}" \
    WHALE_FWD_TMA_WARPSPEC=1 \
        ${PY} "${LEG_DIR}/bench_stable.py" \
        --shapes "${SHAPE_LONG}" \
        --backends "whale,fa3" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label "phaseC_tma_ws_${SAFE_TAG}" \
        --out "${LEG_DIR}/evidence/phaseC_tma_ws_${SAFE_TAG}_${TS}.json" 2>&1 | tee -a "${LOG}"
done

echo "" | tee -a "${LOG}"
echo "== done. log: ${LOG}"
