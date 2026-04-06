#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}/.."

export SEED="${SEED:-300}"
export QUANT_ATTN_BITS="${QUANT_ATTN_BITS:-5}"
export QUANT_MLP_BITS="${QUANT_MLP_BITS:-6}"
export QUANT_AUX_BITS="${QUANT_AUX_BITS:-6}"
export QUANT_EMBED_BITS="${QUANT_EMBED_BITS:-8}"
export QUANT_OTHER_BITS="${QUANT_OTHER_BITS:-8}"
export QUANT_ARTIFACT_PATH="${QUANT_ARTIFACT_PATH:-final_model.attn5_mlp6_aux6_embed8_other8.ptz}"
export RUN_ID="${RUN_ID:-rascal_mixedint_attn5_mlp6_aux6_embed8_other8_seed${SEED}}"

exec bash test_lab/Rascal_II_mixed_int_lab/run.sh "$@"
