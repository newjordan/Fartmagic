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
export QUANT_ATTN_BITS="${QUANT_ATTN_BITS:-5}"
export QUANT_MLP_BITS="${QUANT_MLP_BITS:-6}"
export QUANT_AUX_BITS="${QUANT_AUX_BITS:-6}"
export QUANT_EMBED_BITS="${QUANT_EMBED_BITS:-8}"
export QUANT_OTHER_BITS="${QUANT_OTHER_BITS:-8}"
export QUANT_ARTIFACT_PATH="final_model.arm_b_mu_centering.ptz"
export RUN_ID="ablation_arm_b_mu_centering_seed${SEED}"

TRAIN_SCRIPT="$(find "${REPO_ROOT}" -path "*/Rascal_II_mixed_int_lab/train_gpt_arm_b_mu_centering.py" | head -1)"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
