#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

# Rascal III — Lucky V base (SLOT stripped) + mixed-int (attn=5, embed=8) + brotli-11
# Changes vs Rascal II: brotli compression, QK_GAIN_INIT=5.0, MUON_BACKEND_STEPS=7, mixed-int quant, n-gram eval
export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MAX_WALLCLOCK_SECONDS=600
export RUN_ID="rascal_iii_seed${SEED}"

pip install brotli -q 2>/dev/null || true

TRAIN_SCRIPT="$(find "${REPO_ROOT}" -path "*/Rascal_III/train_gpt.py" | head -1)"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
