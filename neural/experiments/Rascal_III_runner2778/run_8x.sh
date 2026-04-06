#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

# Canonical runner for the legacy 27.78M-parameter Rascal III script
# copied from junkyard into a stable neural/ path.
export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export RUN_ID="${RUN_ID:-rascal_iii_runner2778_seed${SEED}}"

# Pin loader profile to the known-good run behavior.
export LOADER_MODE="coprime"
export COPRIME_SHARDS_PER_BATCH="1"
export COPRIME_SHARD_HOLD_STEPS="64"

# Disable post-train TTT by default so pre-eval model quality is preserved.
export TTT_EPOCHS="${TTT_EPOCHS:-0}"
export TTT_LR="${TTT_LR:-0.0}"
export TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-0}"

TRAIN_SCRIPT="${REPO_ROOT}/neural/experiments/Rascal_III_runner2778/train_gpt.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "ERROR: missing runner script at ${TRAIN_SCRIPT}" >&2
  exit 1
fi

echo "rascal_iii_runner2778_train_script:${TRAIN_SCRIPT}"
echo "rascal_iii_runner2778_profile loader=${LOADER_MODE} shards_per_batch=${COPRIME_SHARDS_PER_BATCH} ttt_epochs=${TTT_EPOCHS} ttt_lr=${TTT_LR} ttt_freeze_blocks=${TTT_FREEZE_BLOCKS}"

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"

