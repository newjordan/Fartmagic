#!/usr/bin/env bash
# whale_dkdv_autotune_expand — bench_stable sweep over 4 shapes.
#
# Preconditions (user must verify before launch):
#   1) vault_patch.md has been applied to vault/whale_kernel_triton.py
#      (BEFORE/AFTER in this leg). _bwd_kv_configs() now returns 56 configs.
#   2) Pod stack = cu130 / PyTorch 2.11.0+cu130 (pod_stack.lock enforced).
#   3) Running on 1xH100 (CUDA_VISIBLE_DEVICES=0).
#
# Modeled on legs/2026-04-16_whale_bench_queue/queue_early_exit_headline.sh.

set -uo pipefail

LEG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${LEG_DIR}/../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${REPO_ROOT}"
source "${LEG_DIR}/tracked_env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY=/venv/main/bin/python3

mkdir -p "${LEG_DIR}/logs" "${LEG_DIR}/evidence"
LOG="${LEG_DIR}/logs/run_${TS}.log"
echo "== ${TS}  whale_dkdv_autotune_expand" | tee "${LOG}"
echo "   WHALE_BWD_VARIANT=${WHALE_BWD_VARIANT}" | tee -a "${LOG}"
echo "   shapes=${WHALE_SHAPES}" | tee -a "${LOG}"
echo "   rounds=${WHALE_BENCH_ROUNDS} iters=${WHALE_BENCH_ITERS}" | tee -a "${LOG}"

# Sanity: assert the config count picked up the expansion.
${PY} -c "from vault.whale_kernel_triton import _bwd_kv_configs; n=len(_bwd_kv_configs()); print(f'_bwd_kv_configs count = {n}'); assert n == 56, f'expected 56 configs after vault_patch; got {n}'" 2>&1 | tee -a "${LOG}"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "[FATAL] vault_patch.md has not been applied — aborting." | tee -a "${LOG}"
    exit 2
fi

# bench_stable.py re-used from the H6 long-T leg (not changing it here).
BENCH="${REPO_ROOT}/legs/2026-04-16_whale_long_t_profile/bench_stable.py"
test -f "${BENCH}" || { echo "[FATAL] missing bench: ${BENCH}" | tee -a "${LOG}"; exit 3; }

IFS=';' read -ra SHAPES <<< "${WHALE_SHAPES}"
for shape in "${SHAPES[@]}"; do
    # Clean triton cache so autotune re-runs across the expanded space
    # for this shape.
    rm -rf /root/.triton/cache 2>/dev/null || true

    SAFE=$(echo "${shape}" | tr ',' '_')
    OUT="${LEG_DIR}/evidence/expand_${SAFE}_${TS}.json"
    echo "" | tee -a "${LOG}"
    echo "==== shape=${shape}  -> ${OUT}" | tee -a "${LOG}"
    ${PY} "${BENCH}" \
        --shape "${shape}" \
        --backends "whale_fast,fa3" \
        --rounds "${WHALE_BENCH_ROUNDS}" \
        --iters  "${WHALE_BENCH_ITERS}" \
        --label "expand_${SAFE}" \
        --out "${OUT}" 2>&1 | tee -a "${LOG}" \
        || echo "[WARN] shape=${shape} FAILED — continuing" | tee -a "${LOG}"
done

echo "" | tee -a "${LOG}"
echo "== done. log: ${LOG}" | tee -a "${LOG}"
echo "== evidence dir: ${LEG_DIR}/evidence" | tee -a "${LOG}"
