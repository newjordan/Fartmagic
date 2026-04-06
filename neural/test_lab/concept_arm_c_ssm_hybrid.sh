#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

# Concept Arm C — SSM Hybrid (first 4 layers SSM, last 7 attention)
export SEED="${SEED:-300}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=9999
export SSM_HYBRID=1
export SSM_LAYERS=4
export SSM_STATE_DIM=64
# Sequential scan loop breaks torch.compile fullgraph mode
export COMPILE_FULLGRAPH=0
export RUN_ID="concept_arm_c_ssm_hybrid_seed${SEED}"

pip install brotli -q 2>/dev/null || true

TRAIN_SCRIPT="$(find "${REPO_ROOT}" -path "*/arm_c_ssm_hybrid/train_gpt.py" | head -1)"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
