#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

# Rascal III — Rascal II base + mixed-int (attn=5, embed=8) + brotli-11 + byte-shuffle
# ONLY confirmed improvements. No Lucky V baggage.
export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MAX_WALLCLOCK_SECONDS=600
export RUN_ID="rascal_iii_seed${SEED}"
# Force canonical Rascal-II loader behavior (avoid inheriting stale shell env vars).
export LOADER_MODE="coprime"
export COPRIME_SHARDS_PER_BATCH="1"
export COPRIME_SHARD_HOLD_STEPS="64"

pip install brotli -q 2>/dev/null || true

TRAIN_SCRIPT="${REPO_ROOT}/neural/experiments/Rascal_III/train_gpt.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  TRAIN_SCRIPT="${REPO_ROOT}/experiments/Rascal_III/train_gpt.py"
fi
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
