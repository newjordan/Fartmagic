#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
pip install brotli zstandard 2>/dev/null || true
python3 -c "import brotli" || { echo "FATAL: brotli not installed"; exit 1; }
SEED="${SEED:-444}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
MAX_WALLCLOCK_SECONDS=600 \
SKIP_GPTQ=1 \
COMPRESSOR="${COMPRESSOR:-brotli}" \
LOADER_MODE=coprime \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "$SCRIPT_DIR/train_gpt.py"
