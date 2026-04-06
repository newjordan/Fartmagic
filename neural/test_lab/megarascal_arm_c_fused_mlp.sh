#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

# Megarascal Arm C — Fused RMSNorm+UpProj+LeakyReLU²+DownProj
export SEED="${SEED:-300}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=9999
export MLP_KERNEL_MODE="fused_norm_act"
export RUN_ID="megarascal_arm_c_fused_mlp_seed${SEED}"

TRAIN_SCRIPT="$(find "${REPO_ROOT}" -path "*/megarascal/train_gpt.py" | head -1)"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
