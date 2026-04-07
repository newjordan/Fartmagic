#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SCRIPT_PATH="${REPO_ROOT}/neural/experiments/1.110_15.5mb_baseline.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
  echo "ERROR: missing baseline file: ${SCRIPT_PATH}" >&2
  exit 1
fi

export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

# Pinned stable profile.
export SKIP_GPTQ=1
export LOADER_MODE=coprime
export COPRIME_MAX_LOADED_SHARDS=1
export COPRIME_SHARDS_PER_BATCH=1
export COPRIME_SHARD_HOLD_STEPS=64
export COMPLEMENT_ALPHA=0
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2048
export ROPE_DIMS=16
export SWA_EVERY=50
export MTP_NUM_HEADS=0
export TRIGRAM=0
export NGRAM_EVAL_ORDER=0
export CUBRIC_CADENCE=0
export NGRAM_ENTROPY_SHIFT=0

echo "rascal_ii_1110_baseline_script:${SCRIPT_PATH}"
echo "rascal_ii_1110_baseline_profile seed=${SEED} nproc=${NPROC_PER_NODE} skip_gptq=${SKIP_GPTQ} loader=${LOADER_MODE} ngram_order=${NGRAM_EVAL_ORDER}"

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_PATH}" "$@"
