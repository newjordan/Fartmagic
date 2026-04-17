#!/usr/bin/env bash
# Kernel-leg gate. Runs numerics + speed + compile smoke tests for the
# whale attention Triton kernels. This does NOT launch training.
#
# Real training is exercised by the sibling leg
# `legs/2026-04-16_whale_pr1493_triton_kernel/gate.sh`, which imports
# the same `vault/whale_kernel_triton.py` that we edit here.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LEG_DIR="${REPO_ROOT}/legs/2026-04-16_whale_fast_kernels"
LOG_DIR="${LEG_DIR}/logs"
EVID_DIR="${LEG_DIR}/evidence"
mkdir -p "${LOG_DIR}" "${EVID_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/gate_${TS}.log"

PY="${PYTHON:-python3}"

echo "[gate] device check"
${PY} -c "import torch; assert torch.cuda.is_available(); print('device:', torch.cuda.get_device_name(0))"

echo "[gate] numerics + speed (current kernel)"
PYTHONPATH="${REPO_ROOT}" ${PY} "${LEG_DIR}/bench_numerics.py" \
  --out "${EVID_DIR}/gate_${TS}_numerics.json" \
  --label "gate_${TS}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[gate] torch.compile smoke"
PYTHONPATH="${REPO_ROOT}" ${PY} "${LEG_DIR}/test_compile.py" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[gate] whale_attention_bench (custom backend)"
PYTHONPATH="${REPO_ROOT}" ${PY} "${REPO_ROOT}/scripts/whale_attention_bench.py" \
  --backend custom \
  --custom-kernel vault.whale_kernel_triton:custom_whale_attn_fwd \
  --suite core --batch-size 8 --seq-len 2048 \
  --num-heads 8 --num-kv-heads 4 --model-dims 512 \
  --warmup-iters 5 --timed-iters 20 \
  --json-out "${EVID_DIR}/gate_${TS}_bench_custom.json" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[gate] whale_attention_bench (sdpa backend, reference)"
PYTHONPATH="${REPO_ROOT}" ${PY} "${REPO_ROOT}/scripts/whale_attention_bench.py" \
  --backend sdpa \
  --suite core --batch-size 8 --seq-len 2048 \
  --num-heads 8 --num-kv-heads 4 --model-dims 512 \
  --warmup-iters 5 --timed-iters 20 \
  --json-out "${EVID_DIR}/gate_${TS}_bench_sdpa.json" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[gate] all kernel-level checks passed. Log: ${LOG_FILE}"
