#!/usr/bin/env bash
# Bench the whale dkdv kernel before/after toggling the existing
# WHALE_BWD_SPLIT_H=1 lever (no vault edit required) on the same 4
# production shapes as Lever A plus 2 launch-bound synthetic shapes.
#
# Usage:
#   bash legs/2026-04-16_whale_dkdv_gqa_grid/run.sh before
#   bash legs/2026-04-16_whale_dkdv_gqa_grid/run.sh after
#
# Outputs land in evidence/ and logs/ under this leg directory.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LEG_DIR="${REPO_ROOT}/legs/2026-04-16_whale_dkdv_gqa_grid"
LOG_DIR="${LEG_DIR}/logs"
EVID_DIR="${LEG_DIR}/evidence"
mkdir -p "${LOG_DIR}" "${EVID_DIR}"

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${LEG_DIR}/tracked_env.sh"

PHASE="${1:-before}"
case "${PHASE}" in
  before|after) ;;
  *) echo "[run] phase must be 'before' or 'after', got '${PHASE}'" >&2; exit 2 ;;
esac

# This is the lever this leg toggles. before = upstream (no split-H);
# after = WHALE_BWD_SPLIT_H=1 (existing kernel + post-kernel reduce).
case "${PHASE}" in
  before) export WHALE_BWD_SPLIT_H=0 ;;
  after)  export WHALE_BWD_SPLIT_H=1 ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${PHASE}_${TS}.log"
PY="${PYTHON:-python3}"

echo "[run] phase=${PHASE} ts=${TS} WHALE_BWD_SPLIT_H=${WHALE_BWD_SPLIT_H} WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT}" \
  | tee "${LOG_FILE}"
echo "[run] preflight"  | tee -a "${LOG_FILE}"
{
  echo "  pwd: $(pwd)"
  echo "  branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "  head: $(git rev-parse --short HEAD)"
  echo "  vault hash: $(sha256sum vault/whale_kernel_triton.py | awk '{print $1}')"
} | tee -a "${LOG_FILE}"

if [[ "${WHALE_FLUSH_TRITON_CACHE:-0}" == "1" ]]; then
  echo "[run] flushing triton autotune cache"  | tee -a "${LOG_FILE}"
  rm -rf "${HOME}/.triton/cache" 2>/dev/null || true
fi

echo "[run] device check" | tee -a "${LOG_FILE}"
${PY} -c "import torch; assert torch.cuda.is_available(); print('device:', torch.cuda.get_device_name(0))" \
  | tee -a "${LOG_FILE}"

# Numerics + micro-speed sweep, reusing the harness from whale_fast_kernels.
NUMERICS_OUT="${EVID_DIR}/${PHASE}_${TS}_numerics.json"
echo "[run] numerics + micro-speed -> ${NUMERICS_OUT}" | tee -a "${LOG_FILE}"
PYTHONPATH="${REPO_ROOT}" ${PY} \
  "${REPO_ROOT}/legs/2026-04-16_whale_fast_kernels/bench_numerics.py" \
  --out "${NUMERICS_OUT}" \
  --label "dkdv_gqa_grid_${PHASE}_${TS}" \
  2>&1 | tee -a "${LOG_FILE}"

# Per-shape backend bench: 4 production shapes from Lever A + 2 launch-bound.
IFS=';' read -r -a SHAPES <<< "${BENCH_SHAPES}"
for shape in "${SHAPES[@]}"; do
  IFS=',' read -r B T H KV D <<< "${shape}"
  TAG="B${B}_T${T}_H${H}_KV${KV}_D${D}"
  CUSTOM_OUT="${EVID_DIR}/${PHASE}_${TS}_${TAG}_custom.json"
  SDPA_OUT="${EVID_DIR}/${PHASE}_${TS}_${TAG}_sdpa.json"
  echo "[run] shape=${TAG} backend=custom -> ${CUSTOM_OUT}" | tee -a "${LOG_FILE}"
  PYTHONPATH="${REPO_ROOT}" ${PY} "${REPO_ROOT}/scripts/whale_attention_bench.py" \
    --backend custom \
    --custom-kernel vault.whale_kernel_triton:custom_whale_attn_fwd \
    --suite core --batch-size "${B}" --seq-len "${T}" \
    --num-heads "${H}" --num-kv-heads "${KV}" --model-dims "$((H * D))" \
    --warmup-iters 5 --timed-iters 50 \
    --json-out "${CUSTOM_OUT}" \
    2>&1 | tee -a "${LOG_FILE}"

  echo "[run] shape=${TAG} backend=sdpa -> ${SDPA_OUT}" | tee -a "${LOG_FILE}"
  PYTHONPATH="${REPO_ROOT}" ${PY} "${REPO_ROOT}/scripts/whale_attention_bench.py" \
    --backend sdpa \
    --suite core --batch-size "${B}" --seq-len "${T}" \
    --num-heads "${H}" --num-kv-heads "${KV}" --model-dims "$((H * D))" \
    --warmup-iters 5 --timed-iters 50 \
    --json-out "${SDPA_OUT}" \
    2>&1 | tee -a "${LOG_FILE}"
done

echo "[run] phase=${PHASE} done. log=${LOG_FILE}" | tee -a "${LOG_FILE}"
