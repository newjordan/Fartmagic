#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PATH="/venv/main/bin:${PATH}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/launch_guard.sh"

LEG_DIR="${REPO_ROOT}/legs/2026-04-20_wh_kern_int_primary_donor"
TRAIN_SCRIPT="${REPO_ROOT}/gemini_runs/wh_kern_int_train_gpt.py"
TRACKED_ENV="${LEG_DIR}/tracked_env.sh"
LOG_DIR="${LEG_DIR}/logs"
ART_DIR="${REPO_ROOT}/artifacts/wh_kern_int_primary_donor"
mkdir -p "${LOG_DIR}" "${ART_DIR}"
LOG_FILE="${LOG_DIR}/full_seed${SEED:-444}_$(date +%Y%m%d_%H%M%S).log"

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
  )
  local name=""
  for name in "${vars[@]}"; do
    if [[ -n "${!name+x}" ]]; then
      die "refusing ad-hoc env override: ${name} is already set in the shell. Edit ${TRACKED_ENV} instead."
    fi
  done
}

reject_adhoc_env
# shellcheck disable=SC1090
source "${TRACKED_ENV}"

lg_preflight_and_lock "${REPO_ROOT}" "${TRAIN_SCRIPT}" "${TRACKED_ENV}" "${NPROC}" "$0"

echo "wh_kern_int run: donor=whale_l12_dim576_matrixbits5 family=${WH_KERN_WINNER_FAMILY_ID} qmr=${WH_KERN_WINNER_QMR}" | tee -a "${LOG_FILE}"
echo "wh_kern_int kernel: dq=${WHALE_BWD_Q_CONFIG} dkdv=${WHALE_BWD_KV_TMA_CONFIG} mr=${WHALE_BWD_KV_TMA_MAXNREG}" | tee -a "${LOG_FILE}"

SEED="${SEED}" \
NPROC_PER_NODE="${NPROC}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
/venv/main/bin/torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
2>&1 | tee -a "${LOG_FILE}"

if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
  cp -f "${REPO_ROOT}/final_model.pt" "${ART_DIR}/final_model_wh_kern_int_primary_donor.pt"
fi
if [[ -f "${REPO_ROOT}/final_model.int6.ptz" ]]; then
  cp -f "${REPO_ROOT}/final_model.int6.ptz" "${ART_DIR}/final_model_wh_kern_int_primary_donor.int6.ptz"
fi
if [[ -f "${REPO_ROOT}/final_model.int8.ptz" ]]; then
  cp -f "${REPO_ROOT}/final_model.int8.ptz" "${ART_DIR}/final_model_wh_kern_int_primary_donor.int8.ptz"
fi

echo "LOG: ${LOG_FILE}"
