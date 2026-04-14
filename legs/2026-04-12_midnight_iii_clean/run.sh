#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/launch_guard.sh"

TRAIN_SCRIPT="${REPO_ROOT}/legs/2026-04-12_midnight_iii_clean/train_gpt.py"
LOG_DIR="${REPO_ROOT}/legs/2026-04-12_midnight_iii_clean/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/full_seed${SEED:-444}_$(date +%Y%m%d_%H%M%S).log"

NPROC="${NPROC_PER_NODE:-8}"
SEED="${SEED:-444}"
TRACKED_ENV="${REPO_ROOT}/legs/2026-04-12_midnight_iii_clean/tracked_env.sh"

die() {
  echo "FATAL: $*" >&2
  exit 1
}

reject_adhoc_env() {
  local vars=(
    MAX_WALLCLOCK_SECONDS SKIP_GPTQ
    COMPRESSOR NUM_LAYERS QUANT_ATTN_BITS QUANT_MLP_BITS QUANT_AUX_BITS
    QUANT_EMBED_BITS QUANT_OTHER_BITS LOADER_MODE COPRIME_MAX_LOADED_SHARDS
    COPRIME_SHARDS_PER_BATCH COPRIME_SHARD_HOLD_STEPS COMPLEMENT_ALPHA
    XSA_LAST_N BIGRAM_VOCAB_SIZE ROPE_DIMS SWA_EVERY MTP_NUM_HEADS
    TRIGRAM NGRAM_EVAL_ORDER CUBRIC_CADENCE NGRAM_ENTROPY_SHIFT
    VOCAB_SIZE DATA_PATH TOKENIZER_PATH VE_ENABLED NUM_LOOPS
    QK_GAIN_INIT MATRIX_LR MUON_WD
  )
  local name=""
  for name in "${vars[@]}"; do
    if [[ -n "${!name+x}" ]]; then
      die "refusing ad-hoc env override: ${name} is already set in the shell. Edit ${TRACKED_ENV} instead."
    fi
  done
}

# Mandatory preflight: trainer diff must pass before the full run starts.
# For intentional framework-level legs, pass explicit overrides, e.g.:
# LEG_DIFF_GUARD_ARGS="--max-code-changes 80 --max-total-changed-lines 120" bash ${REPO_ROOT}/legs/2026-04-12_midnight_iii_clean/run.sh
if [[ -n "${LEG_DIFF_GUARD_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python3 "${REPO_ROOT}/scripts/leg_diff_guard.py" "${REPO_ROOT}/legs/2026-04-12_midnight_iii_clean" ${LEG_DIFF_GUARD_ARGS}
else
  python3 "${REPO_ROOT}/scripts/leg_diff_guard.py" "${REPO_ROOT}/legs/2026-04-12_midnight_iii_clean"
fi

reject_adhoc_env
# shellcheck disable=SC1090
source "${TRACKED_ENV}"

lg_preflight_and_lock "${REPO_ROOT}" "${TRAIN_SCRIPT}" "${TRACKED_ENV}" "${NPROC}" "$0"

SEED="${SEED}" \
NPROC_PER_NODE="${NPROC}" \
MAX_WALLCLOCK_SECONDS=600 \
SKIP_GPTQ=1 \
torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
2>&1 | tee "${LOG_FILE}"

echo "LOG: ${LOG_FILE}"
