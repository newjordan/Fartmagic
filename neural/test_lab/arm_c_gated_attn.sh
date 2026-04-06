#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

export SEED="${SEED:-300}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export GATED_ATTENTION=1
export QUANT_ATTN_BITS="${QUANT_ATTN_BITS:-5}"
export QUANT_MLP_BITS="${QUANT_MLP_BITS:-6}"
export QUANT_AUX_BITS="${QUANT_AUX_BITS:-6}"
export QUANT_EMBED_BITS="${QUANT_EMBED_BITS:-8}"
export QUANT_OTHER_BITS="${QUANT_OTHER_BITS:-8}"
export QUANT_ARTIFACT_PATH="final_model.arm_c_gated_attn.ptz"
export RUN_ID="ablation_arm_c_gated_attn_seed${SEED}"

exec bash test_lab/Rascal_II_mixed_int_lab/run.sh "$@"
