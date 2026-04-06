#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

# Rascal IV campaign runner.
# Arms:
# - control: hardened Rascal III profile cloned into Rascal IV
# - mtp1: control + MTP_NUM_HEADS=1
# - longctx: control + TRAIN/EVAL seq len 4096
# - e2e_ttt: concept arm with TTT_E2E fast MLP layers (separate train script)
ARM="${RASCAL_IV_ARM:-control}"
ALLOW_EXPERIMENTAL="${RASCAL_IV_ALLOW_EXPERIMENTAL:-0}"

export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export RUN_ID="${RUN_ID:-rascal_iv_${ARM}_seed${SEED}}"

# Canonical loader behavior.
export LOADER_MODE="coprime"
export COPRIME_SHARDS_PER_BATCH="1"
export COPRIME_SHARD_HOLD_STEPS="64"

# Canonical non-Lucky defaults.
export QK_GAIN_INIT="1.5"
export MUON_BACKEND_STEPS="5"
export MTP_NUM_HEADS="0"

# Keep risky/failed toggles disabled by default.
export TRIGRAM="0"
export GATED_ATTENTION="0"
export VALUE_RESIDUAL="0"
export DTG_ENABLED="0"
export QAT_ENABLED="0"

# Canonical mixed-int export profile.
export QUANT_ATTN_BITS="5"
export QUANT_MLP_BITS="6"
export QUANT_AUX_BITS="6"
export QUANT_EMBED_BITS="8"
export QUANT_OTHER_BITS="8"
export QUANT_ARTIFACT_PATH="${QUANT_ARTIFACT_PATH:-final_model.rascal_iv_${ARM}_seed${SEED}.ptz}"

# Guardrails against inherited shell state.
export SKIP_GPTQ="${SKIP_GPTQ:-0}"
export COMPILE_MODE="${COMPILE_MODE:-}"
export MLP_KERNEL_MODE="${MLP_KERNEL_MODE:-}"
export NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-0}"
export NGRAM_EVAL_MAX_SECONDS="${NGRAM_EVAL_MAX_SECONDS:-0}"
export NGRAM_ENTROPY_SHIFT="${NGRAM_ENTROPY_SHIFT:-0}"
export QUANT_ROUNDTRIP_EVAL="${QUANT_ROUNDTRIP_EVAL:-1}"
export EXPORT_RESERVE_MS="${EXPORT_RESERVE_MS:-30000}"

# Hard-disable legacy eval-time adaptation paths (TTT/SLOT variants) if inherited
# from old shells or other repos. Current Rascal IV train script ignores most of
# these, but pinning them prevents accidental cross-run contamination.
export TTT_ENABLED="0"
export TTT_EPOCHS="0"
export TTT_LR="0.0"
export TTT_CHUNK_TOKENS="32768"
export TTT_FREEZE_BLOCKS="0"
export SCALE_TTT_ENABLED="0"
export SLOT_ENABLED="0"
export TTT_E2E="0"

TRAIN_SCRIPT="${REPO_ROOT}/neural/experiments/Rascal_IV/train_gpt.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  TRAIN_SCRIPT="${REPO_ROOT}/experiments/Rascal_IV/train_gpt.py"
fi

if [ "${ARM}" != "control" ] && [ "${ALLOW_EXPERIMENTAL}" != "1" ]; then
  echo "ERROR: refusing experimental arm '${ARM}' without RASCAL_IV_ALLOW_EXPERIMENTAL=1" >&2
  echo "Run control, or explicitly opt in for ablations." >&2
  exit 1
fi

case "${ARM}" in
  control)
    ;;
  mtp1)
    export MTP_NUM_HEADS="1"
    ;;
  longctx)
    export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-4096}"
    export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-4096}"
    ;;
  e2e_ttt)
    export TTT_E2E="1"
    export TTT_HIDDEN="${TTT_HIDDEN:-64}"
    TRAIN_SCRIPT="$(find "${REPO_ROOT}" -path "*/concept_ssm_ttt/arm_b_e2e_ttt/train_gpt.py" | head -1)"
    if [ -z "${TRAIN_SCRIPT}" ] || [ ! -f "${TRAIN_SCRIPT}" ]; then
      echo "ERROR: could not find concept arm_b_e2e_ttt/train_gpt.py" >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: unknown RASCAL_IV_ARM='${ARM}' (expected: control|mtp1|longctx|e2e_ttt)" >&2
    exit 1
    ;;
esac

pip install brotli -q 2>/dev/null || true

echo "rascal_iv_profile arm=${ARM} loader=${LOADER_MODE} shards_per_batch=${COPRIME_SHARDS_PER_BATCH} qk_gain=${QK_GAIN_INIT} muon_backend_steps=${MUON_BACKEND_STEPS} skip_gptq=${SKIP_GPTQ} export_reserve_ms=${EXPORT_RESERVE_MS} ngram_order=${NGRAM_EVAL_ORDER} quant_roundtrip_eval=${QUANT_ROUNDTRIP_EVAL} ttt_enabled=${TTT_ENABLED} ttt_e2e=${TTT_E2E} slot_enabled=${SLOT_ENABLED}"

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
