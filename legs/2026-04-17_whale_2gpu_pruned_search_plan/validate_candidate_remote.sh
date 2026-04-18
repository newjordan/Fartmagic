#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${1:?ssh host required}"
SSH_PORT="${2:?ssh port required}"
DKDV_TILE="${3:?dkdv tile required}"
DKDV_MAXNREG="${4:?dkdv maxnreg required}"
DQ_TILE="${5:?dq tile required}"
DQ_MAXNREG="${6:?dq maxnreg required}"
GPU_ID="${7:-0}"
SSH_KEY="/home/frosty40/.ssh/id_ed25519_apollo"
REMOTE_ROOT="/workspace/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_plan"
STAMP="$(date +%Y%m%d_%H%M%S)"

/usr/bin/ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p "${SSH_PORT}" root@"${SSH_HOST}" bash -lc "
  set -euo pipefail
  cd ${REMOTE_ROOT}
  export CUDA_VISIBLE_DEVICES=${GPU_ID}
  export TRITON_CACHE_DIR=/tmp/triton_cache_whale_validate_${STAMP}
  export PYTHONPATH=${REMOTE_ROOT}/runtime/validate_${STAMP}:\${PYTHONPATH:-}
  mkdir -p runtime/validate_${STAMP}/frozen_vault logs evidence /tmp/triton_cache_whale_validate_${STAMP}
  cp bench_stable_scratch.py runtime/validate_${STAMP}/bench_stable_scratch.py
  cp frozen_vault/__init__.py runtime/validate_${STAMP}/frozen_vault/__init__.py
  cp frozen_vault/whale_kernel_triton.py runtime/validate_${STAMP}/frozen_vault/whale_kernel_triton.py
  export WHALE_BWD_VARIANT=fused_delta_tma
  export WHALE_FUSED_DELTA_T_MAX=3072
  export WHALE_BWD_KV_TMA_CONFIG='${DKDV_TILE}'
  export WHALE_BWD_KV_TMA_MAXNREG='${DKDV_MAXNREG}'
  export WHALE_BWD_Q_CONFIG='${DQ_TILE}'
  export WHALE_BWD_Q_MAXNREG='${DQ_MAXNREG}'
  nohup /venv/main/bin/python3 runtime/validate_${STAMP}/bench_stable_scratch.py     --label validate_${STAMP}     --shape 2,2048,8,4,64     --backends whale_fast,fa3     --rounds 10     --iters 200     --out evidence/validate_${STAMP}.json > logs/validate_${STAMP}.log 2>&1 & echo PID=\$!"
