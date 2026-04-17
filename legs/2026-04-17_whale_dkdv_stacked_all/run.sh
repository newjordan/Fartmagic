#!/usr/bin/env bash
# Coordination bench for the 4-lever whale dkdv stack.
# Each phase assumes its corresponding vault patch has already landed.
# This script does NOT patch vault/whale_kernel_triton.py.
#
# Phases (see plan.md):
#   phase0_baseline     — measure current fb on 4 shapes (no vault edits)
#   phase1_autotune     — Lever A landed; bench
#   phase2_masksplit    — Levers A+0 landed; bench
#   phase3_tma          — Levers A+0+B landed; WHALE_DKDV_TMA_LOADS=1
#   phase4_persistent   — Levers A+0+B+C landed; WHALE_DKDV_PERSISTENT=1
#
# Usage:
#   bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase0_baseline
#   bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase1_autotune
#   bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase2_masksplit
#   WHALE_DKDV_TMA_LOADS=1 \
#     bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase3_tma
#   WHALE_DKDV_PERSISTENT=1 WHALE_DKDV_TMA_LOADS=1 \
#     bash legs/2026-04-17_whale_dkdv_stacked_all/run.sh phase4_persistent
#
# Outputs land in evidence/ and logs/ under this leg directory.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LEG_DIR="${REPO_ROOT}/legs/2026-04-17_whale_dkdv_stacked_all"
LOG_DIR="${LEG_DIR}/logs"
EVID_DIR="${LEG_DIR}/evidence"
mkdir -p "${LOG_DIR}" "${EVID_DIR}"

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${LEG_DIR}/tracked_env.sh"

PHASE="${1:-phase0_baseline}"
case "${PHASE}" in
  phase0_baseline|phase1_autotune|phase2_masksplit|phase3_tma|phase4_persistent) ;;
  *) echo "[run] phase must be one of: phase0_baseline, phase1_autotune, phase2_masksplit, phase3_tma, phase4_persistent. got '${PHASE}'" >&2
     exit 2 ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${PHASE}_${TS}.log"
PY="${PYTHON:-python3}"

# --- Preflight (per CLAUDE.md startup safety protocol) ---
{
  echo "[run] phase=${PHASE} ts=${TS}"
  echo "[preflight] pwd: $(pwd)"
  echo "[preflight] branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "[preflight] head: $(git rev-parse --short HEAD)"
  echo "[preflight] vault sha256: $(sha256sum vault/whale_kernel_triton.py | awk '{print $1}')"
  echo "[preflight] git remote: $(git remote get-url origin 2>/dev/null || echo '<none>')"
  echo "[env] WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT}"
  echo "[env] WHALE_FUSED_DELTA_T_MAX=${WHALE_FUSED_DELTA_T_MAX}"
  echo "[env] WHALE_DKDV_TMA_LOADS=${WHALE_DKDV_TMA_LOADS}"
  echo "[env] WHALE_DKDV_PERSISTENT=${WHALE_DKDV_PERSISTENT}"
  echo "[env] WHALE_FLUSH_TRITON_CACHE=${WHALE_FLUSH_TRITON_CACHE}"
  echo "[env] BENCH_SHAPES=${BENCH_SHAPES}"
} | tee "${LOG_FILE}"

# Preflight required files
for f in vault/whale_kernel_triton.py scripts/whale_attention_bench.py; do
  if [[ ! -f "${REPO_ROOT}/${f}" ]]; then
    echo "[preflight] MISSING required file: ${f}" | tee -a "${LOG_FILE}" >&2
    exit 3
  fi
done

# Optional numerics harness (same script used by early-exit leg)
NUMERICS_SCRIPT="${REPO_ROOT}/legs/2026-04-16_whale_fast_kernels/bench_numerics.py"
RUN_NUMERICS=0
if [[ -f "${NUMERICS_SCRIPT}" ]]; then
  RUN_NUMERICS=1
fi

if [[ "${WHALE_FLUSH_TRITON_CACHE:-0}" == "1" ]]; then
  echo "[run] flushing triton autotune cache" | tee -a "${LOG_FILE}"
  rm -rf "${HOME}/.triton/cache" 2>/dev/null || true
fi

echo "[run] device check" | tee -a "${LOG_FILE}"
${PY} -c "import torch; assert torch.cuda.is_available(); print('device:', torch.cuda.get_device_name(0))" \
  | tee -a "${LOG_FILE}"

# --- Per-phase env overrides ---
case "${PHASE}" in
  phase0_baseline|phase1_autotune|phase2_masksplit)
    export WHALE_DKDV_TMA_LOADS=0
    export WHALE_DKDV_PERSISTENT=0
    ;;
  phase3_tma)
    export WHALE_DKDV_TMA_LOADS=1
    export WHALE_DKDV_PERSISTENT=0
    ;;
  phase4_persistent)
    # Inherit WHALE_DKDV_TMA_LOADS from caller (plan.md expects =1 here).
    export WHALE_DKDV_PERSISTENT=1
    ;;
esac

echo "[run] effective env: WHALE_DKDV_TMA_LOADS=${WHALE_DKDV_TMA_LOADS} WHALE_DKDV_PERSISTENT=${WHALE_DKDV_PERSISTENT}" \
  | tee -a "${LOG_FILE}"

# --- Numerics pass (once per phase, if harness is present) ---
if [[ "${RUN_NUMERICS}" == "1" ]]; then
  NUMERICS_OUT="${EVID_DIR}/${PHASE}_${TS}_numerics.json"
  echo "[run] numerics -> ${NUMERICS_OUT}" | tee -a "${LOG_FILE}"
  PYTHONPATH="${REPO_ROOT}" ${PY} "${NUMERICS_SCRIPT}" \
    --out "${NUMERICS_OUT}" \
    --label "dkdv_stacked_${PHASE}_${TS}" \
    2>&1 | tee -a "${LOG_FILE}" || {
      echo "[run] NUMERICS FAILED for phase=${PHASE}. Abort." | tee -a "${LOG_FILE}"
      exit 4
    }
else
  echo "[run] numerics harness not present at ${NUMERICS_SCRIPT}; skipping" | tee -a "${LOG_FILE}"
fi

# --- Per-shape bench: custom (whale) and sdpa baseline ---
IFS=';' read -r -a SHAPES <<< "${BENCH_SHAPES}"
for shape in "${SHAPES[@]}"; do
  IFS=',' read -r B T H KV D <<< "${shape}"
  TAG="B${B}_T${T}_H${H}_KV${KV}_D${D}"
  CUSTOM_OUT="${EVID_DIR}/${PHASE}_${TS}_${TAG}_custom.json"
  SDPA_OUT="${EVID_DIR}/${PHASE}_${TS}_${TAG}_sdpa.json"
  FA3_OUT="${EVID_DIR}/${PHASE}_${TS}_${TAG}_fa3.json"

  echo "[run] shape=${TAG} backend=custom -> ${CUSTOM_OUT}" | tee -a "${LOG_FILE}"
  PYTHONPATH="${REPO_ROOT}" ${PY} "${REPO_ROOT}/scripts/whale_attention_bench.py" \
    --backend custom \
    --custom-kernel vault.whale_kernel_triton:custom_whale_attn_fwd \
    --suite core --batch-size "${B}" --seq-len "${T}" \
    --num-heads "${H}" --num-kv-heads "${KV}" --model-dims "$((H * D))" \
    --warmup-iters "${WHALE_BENCH_WARMUP}" --timed-iters "${WHALE_BENCH_TIMED}" \
    --json-out "${CUSTOM_OUT}" \
    2>&1 | tee -a "${LOG_FILE}"

  echo "[run] shape=${TAG} backend=sdpa -> ${SDPA_OUT}" | tee -a "${LOG_FILE}"
  PYTHONPATH="${REPO_ROOT}" ${PY} "${REPO_ROOT}/scripts/whale_attention_bench.py" \
    --backend sdpa \
    --suite core --batch-size "${B}" --seq-len "${T}" \
    --num-heads "${H}" --num-kv-heads "${KV}" --model-dims "$((H * D))" \
    --warmup-iters "${WHALE_BENCH_WARMUP}" --timed-iters "${WHALE_BENCH_TIMED}" \
    --json-out "${SDPA_OUT}" \
    2>&1 | tee -a "${LOG_FILE}"

  # FA3 reference (will no-op if FA3 wheel not importable — the bench
  # script reports fa3_available in its JSON).
  echo "[run] shape=${TAG} backend=fa3 -> ${FA3_OUT}" | tee -a "${LOG_FILE}"
  PYTHONPATH="${REPO_ROOT}" ${PY} "${REPO_ROOT}/scripts/whale_attention_bench.py" \
    --backend fa3 \
    --suite core --batch-size "${B}" --seq-len "${T}" \
    --num-heads "${H}" --num-kv-heads "${KV}" --model-dims "$((H * D))" \
    --warmup-iters "${WHALE_BENCH_WARMUP}" --timed-iters "${WHALE_BENCH_TIMED}" \
    --json-out "${FA3_OUT}" \
    2>&1 | tee -a "${LOG_FILE}" || {
      echo "[run] FA3 backend failed (non-fatal) for shape=${TAG}" | tee -a "${LOG_FILE}"
    }
done

echo "[run] phase=${PHASE} done. log=${LOG_FILE}" | tee -a "${LOG_FILE}"
