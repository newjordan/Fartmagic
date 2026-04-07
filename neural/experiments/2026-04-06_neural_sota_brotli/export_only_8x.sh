#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/experiments/2026-04-06_neural_sota/train_gpt.py"

if [[ -z "${INIT_MODEL_PATH:-}" ]]; then
  echo "ERROR: set INIT_MODEL_PATH to your proven final_model.pt"
  echo "Example:"
  echo "  INIT_MODEL_PATH=/home/frosty40/sota_nueral/records/track_10min_16mb/2026-03-30_Rascal_8xH100/final_model.pt \\"
  echo "  SEED=444 NPROC_PER_NODE=8 COMPRESSOR=brotli SKIP_GPTQ=0 bash experiments/2026-04-06_neural_sota/export_only_8x.sh"
  exit 1
fi

cd "$ROOT"

SEED="${SEED:-444}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
INIT_MODEL_PATH="${INIT_MODEL_PATH}" \
ITERATIONS=0 \
WARMUP_STEPS=0 \
MAX_WALLCLOCK_SECONDS=0 \
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}" \
POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}" \
LOADER_MODE=coprime \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
COMPRESSOR="${COMPRESSOR:-brotli}" \
SKIP_GPTQ="${SKIP_GPTQ:-0}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "$SCRIPT"
