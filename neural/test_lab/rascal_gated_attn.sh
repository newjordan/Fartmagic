#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

# Rascal II Gated Attention — GATED_ATTENTION=1, ONE variable vs baseline
export SEED="${SEED:-300}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=9999
export GATED_ATTENTION=1
export RUN_ID="rascal_gated_attn_seed${SEED}"

TRAIN_SCRIPT="${REPO_ROOT}/vault/1.110_15.5mb_baseline.py"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
