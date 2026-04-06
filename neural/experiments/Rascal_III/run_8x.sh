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
# Force canonical Rascal III behavior (avoid inheriting stale shell env vars).
export LOADER_MODE="coprime"
export COPRIME_SHARDS_PER_BATCH="1"
export COPRIME_SHARD_HOLD_STEPS="64"
export QK_GAIN_INIT="1.5"
export MUON_BACKEND_STEPS="5"
export MTP_NUM_HEADS="0"
export TRIGRAM="0"
export GATED_ATTENTION="0"
export VALUE_RESIDUAL="0"
export DTG_ENABLED="0"
export QAT_ENABLED="0"
export QUANT_ATTN_BITS="5"
export QUANT_MLP_BITS="6"
export QUANT_AUX_BITS="6"
export QUANT_EMBED_BITS="8"
export QUANT_OTHER_BITS="8"
export COMPILE_MODE=""
export MLP_KERNEL_MODE=""

# Extra guard against legacy eval-time adaptation envs.
export TTT_ENABLED="0"
export TTT_EPOCHS="0"
export TTT_LR="0.0"
export TTT_FREEZE_BLOCKS="0"
export SCALE_TTT_ENABLED="0"
export SLOT_ENABLED="0"

pip install brotli -q 2>/dev/null || true

TRAIN_SCRIPT="${REPO_ROOT}/neural/experiments/Rascal_III/train_gpt.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  TRAIN_SCRIPT="${REPO_ROOT}/experiments/Rascal_III/train_gpt.py"
fi
echo "rascal_iii_train_script:${TRAIN_SCRIPT}"
echo "rascal_iii_profile loader=${LOADER_MODE} shards_per_batch=${COPRIME_SHARDS_PER_BATCH} qk_gain=${QK_GAIN_INIT} muon_backend_steps=${MUON_BACKEND_STEPS} quant_bits=${QUANT_ATTN_BITS}/${QUANT_MLP_BITS}/${QUANT_AUX_BITS}/${QUANT_EMBED_BITS}/${QUANT_OTHER_BITS} ttt_enabled=${TTT_ENABLED} slot_enabled=${SLOT_ENABLED}"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
