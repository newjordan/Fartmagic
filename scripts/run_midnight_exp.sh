#!/usr/bin/env bash
# run_midnight_exp.sh — launch any experiment script with Midnight 12L env vars.
# Usage: bash scripts/run_midnight_exp.sh <path_to_train_gpt.py> [SEED] [SKIP_GPTQ]
# Example: bash scripts/run_midnight_exp.sh neural/experiments/2026-04-09_midnight_GPTQ/train_gpt.py 444 0
set -euo pipefail

SCRIPT="${1:?Usage: bash scripts/run_midnight_exp.sh <train_gpt.py> [SEED] [SKIP_GPTQ]}"
SEED="${2:-444}"
SKIP_GPTQ="${3:-1}"
NPROC="${NPROC_PER_NODE:-8}"

[[ -f "${SCRIPT}" ]] || { echo "FATAL: file not found: ${SCRIPT}" >&2; exit 1; }

export PYTHONPATH="${PYTHONPATH:-}"

SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS=600 \
SKIP_GPTQ="${SKIP_GPTQ}" \
COMPRESSOR=brotli \
NUM_LAYERS=12 \
QUANT_ATTN_BITS=5 \
QUANT_MLP_BITS=6 \
QUANT_AUX_BITS=6 \
QUANT_EMBED_BITS=8 \
QUANT_OTHER_BITS=8 \
LOADER_MODE=coprime \
COPRIME_MAX_LOADED_SHARDS=1 \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=0 \
torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT}"
