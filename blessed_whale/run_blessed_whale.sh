#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="/venv/main/bin:${PATH}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/launch_guard.sh"

BLESSED_WHALE_DIR="${REPO_ROOT}/blessed_whale"
TRAIN_SCRIPT="${BLESSED_WHALE_DIR}/blessed_whale_train_gpt.py"
TRACKED_ENV="${BLESSED_WHALE_DIR}/blessed_whale_env.sh"
LOG_DIR="${BLESSED_WHALE_DIR}/logs"
ART_DIR="${REPO_ROOT}/artifacts/blessed_whale"
mkdir -p "${LOG_DIR}" "${ART_DIR}"

NPROC="${NPROC_PER_NODE:-8}"
SEED="${SEED:-444}"

die() {
  echo "FATAL: $*" >&2
  exit 1
}

reject_adhoc_env() {
  local vars=(
    MAX_WALLCLOCK_SECONDS DATA_DIR RUN_ID MODEL_DIM EMBEDDING_DIM MATRIX_BITS
    NUM_LAYERS XSA_LAST_N ROPE_DIMS VOCAB_SIZE NUM_LOOPS LOOP_START LOOP_END
    ENABLE_LOOPING_AT PARALLEL_RESIDUAL_START QK_GAIN_INIT MLP_MULT MATRIX_LR
    MUON_MOMENTUM MUON_WD EMA_DECAY WARMDOWN_FRAC TTT_ENABLED TTT_LR TTT_EPOCHS
    TTT_MOMENTUM WHALE_BWD_VARIANT WHALE_FUSED_DELTA_T_MAX WHALE_BWD_Q_CONFIG
    WHALE_BWD_Q_MAXNREG WHALE_BWD_KV_TMA_CONFIG WHALE_BWD_KV_TMA_MAXNREG
    BLESSED_WHALE_ATTN_MODE WH_KERN_ATTN_MODE
  )
  local name=""
  for name in "${vars[@]}"; do
    if [[ -n "${!name+x}" ]]; then
      die "refusing ad-hoc env override: ${name} is already set in the shell. Edit ${TRACKED_ENV} or set BLESSED_WHALE_BENCHMARK_ID instead."
    fi
  done
}

reject_adhoc_env
# shellcheck disable=SC1090
source "${TRACKED_ENV}"

LOG_FILE="${LOG_DIR}/${BLESSED_WHALE_BENCHMARK_ID,,}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
ARTIFACT_TAG="blessed_whale_${BLESSED_WHALE_BENCHMARK_ID,,}"

lg_preflight_and_lock "${REPO_ROOT}" "${TRAIN_SCRIPT}" "${TRACKED_ENV}" "${NPROC}" "$0"

echo "blessed_whale run: benchmark=${BLESSED_WHALE_BENCHMARK_ID} desc=${BLESSED_WHALE_DESCRIPTION}" | tee -a "${LOG_FILE}"
echo "blessed_whale contract: trainer=${BLESSED_WHALE_TRAINER_CONTRACT} attn_mode=${BLESSED_WHALE_ATTN_MODE}" | tee -a "${LOG_FILE}"
echo "blessed_whale donor: whale_l12_dim576_matrixbits5 family=${BLESSED_WHALE_FAMILY_ID} qmr=${BLESSED_WHALE_QMR}" | tee -a "${LOG_FILE}"
echo "blessed_whale kernel: profile=${BLESSED_WHALE_PROFILE_ID} dq=${WHALE_BWD_Q_CONFIG} dkdv=${WHALE_BWD_KV_TMA_CONFIG} mr=${WHALE_BWD_KV_TMA_MAXNREG}" | tee -a "${LOG_FILE}"

SEED="${SEED}" \
NPROC_PER_NODE="${NPROC}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
/venv/main/bin/torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
2>&1 | tee -a "${LOG_FILE}"

if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
  cp -f "${REPO_ROOT}/final_model.pt" "${ART_DIR}/final_model_${ARTIFACT_TAG}.pt"
fi
if [[ -f "${REPO_ROOT}/final_model.int6.ptz" ]]; then
  cp -f "${REPO_ROOT}/final_model.int6.ptz" "${ART_DIR}/final_model_${ARTIFACT_TAG}.int6.ptz"
fi
if [[ -f "${REPO_ROOT}/final_model.int8.ptz" ]]; then
  cp -f "${REPO_ROOT}/final_model.int8.ptz" "${ART_DIR}/final_model_${ARTIFACT_TAG}.int8.ptz"
fi

echo "LOG: ${LOG_FILE}"
