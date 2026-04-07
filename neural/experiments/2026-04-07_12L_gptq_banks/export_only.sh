#!/usr/bin/env bash
# Re-export a saved checkpoint with bank-aware GPTQ. No training.
# Usage: INIT_MODEL_PATH=/path/to/final_model_12L_s300.pt bash export_only.sh
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
pip install brotli zstandard 2>/dev/null || true
python3 -c "import brotli" || { echo "FATAL: brotli not installed"; exit 1; }

if [[ -z "${INIT_MODEL_PATH:-}" ]]; then
    echo "ERROR: set INIT_MODEL_PATH to your proven final_model.pt"
    echo "Example:"
    echo "  INIT_MODEL_PATH=/workspace/parameter-golf/final_model_12L_s300.pt bash export_only.sh"
    exit 1
fi

export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export ITERATIONS=0
export WARMUP_STEPS=0
export MAX_WALLCLOCK_SECONDS=0
export SKIP_GPTQ=0
export COMPRESSOR="${COMPRESSOR:-brotli}"
export LOADER_MODE=coprime
export COPRIME_SHARDS_PER_BATCH=1
export COPRIME_SHARD_HOLD_STEPS=64
export TRIGRAM=0
export NGRAM_EVAL_ORDER=0
export QUANT_ATTN_BITS="${QUANT_ATTN_BITS:-5}"
export QUANT_MLP_BITS="${QUANT_MLP_BITS:-6}"
export QUANT_EMBED_BITS="${QUANT_EMBED_BITS:-8}"
export NUM_LAYERS=12
export SKIP_FINAL_EVAL=0
export POST_EMA_DIAGNOSTIC=1
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "$SCRIPT_DIR/train_gpt.py"
