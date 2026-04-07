#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

if [ -z "${INIT_MODEL_PATH:-}" ]; then
  echo "ERROR: set INIT_MODEL_PATH to the checkpoint to export." >&2
  echo "Example: INIT_MODEL_PATH=/workspace/parameter-golf/final_model.pt SEED=444 NPROC_PER_NODE=8 bash neural/experiments/Rascal_III_runner2778/export_checkpoint_8x.sh" >&2
  exit 1
fi
if [ ! -f "${INIT_MODEL_PATH}" ] && [ -f "${REPO_ROOT}/${INIT_MODEL_PATH}" ]; then
  INIT_MODEL_PATH="${REPO_ROOT}/${INIT_MODEL_PATH}"
fi
if [ ! -f "${INIT_MODEL_PATH}" ]; then
  echo "ERROR: INIT_MODEL_PATH not found: ${INIT_MODEL_PATH}" >&2
  exit 1
fi
export INIT_MODEL_PATH

export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export RUN_ID="${RUN_ID:-rascal_iii_runner2778_ckpt_export_seed${SEED}}"

# Checkpoint-only mode: no training updates.
export ITERATIONS="${ITERATIONS:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-0}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}"

# Lock to known-good runner defaults.
export LOADER_MODE="${LOADER_MODE:-coprime}"
export COPRIME_SHARDS_PER_BATCH="${COPRIME_SHARDS_PER_BATCH:-1}"
export COPRIME_SHARD_HOLD_STEPS="${COPRIME_SHARD_HOLD_STEPS:-64}"
export TTT_EPOCHS="${TTT_EPOCHS:-0}"
export TTT_LR="${TTT_LR:-0.0}"
export TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-0}"

# Export settings.
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export QUANT_ATTN_BITS="${QUANT_ATTN_BITS:-5}"
export QUANT_MLP_BITS="${QUANT_MLP_BITS:-6}"
export QUANT_AUX_BITS="${QUANT_AUX_BITS:-6}"
export QUANT_EMBED_BITS="${QUANT_EMBED_BITS:-8}"
export QUANT_OTHER_BITS="${QUANT_OTHER_BITS:-8}"
export QUANT_ROUNDTRIP_EVAL="${QUANT_ROUNDTRIP_EVAL:-1}"
export QUANT_ARTIFACT_PATH="${QUANT_ARTIFACT_PATH:-final_model.rascal_iii_runner2778_ckpt_seed${SEED}.ptz}"

# No final sliding/ngram eval in export pass unless explicitly requested.
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"

pip install brotli -q 2>/dev/null || true

# Preflight on CPU to avoid expensive GPU launch with a wrong checkpoint.
python3 - <<'PY'
import os, sys, torch
p = os.environ["INIT_MODEL_PATH"]
obj = torch.load(p, map_location="cpu")
if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
    state = obj["state_dict"]
elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
    state = obj["model"]
elif isinstance(obj, dict):
    state = obj
else:
    print(f"ERROR: INIT_MODEL_PATH is not a state_dict-like dict: {type(obj)}", file=sys.stderr)
    sys.exit(2)
prefixes = ("module.", "_orig_mod.", "model.", "base_model.")
normalized = {}
for k, v in state.items():
    if not isinstance(v, torch.Tensor):
        continue
    nk = k
    changed = True
    while changed:
        changed = False
        for pfx in prefixes:
            if nk.startswith(pfx):
                nk = nk[len(pfx):]
                changed = True
    normalized[nk] = v
required = ["tok_emb.weight", "skip_weights", "bigram.ngram_gate", "qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank"]
missing = [k for k in required if k not in normalized]
if missing:
    print("ERROR: checkpoint does not look like Rascal_III_runner2778 state_dict", file=sys.stderr)
    print("missing_required_keys:", missing, file=sys.stderr)
    sample = sorted(list(normalized.keys()))[:20]
    print("sample_keys:", sample, file=sys.stderr)
    sys.exit(3)
print(f"ckpt_preflight:ok keys={len(normalized)} skip_weights_shape={tuple(normalized['skip_weights'].shape)}")
PY

TRAIN_SCRIPT="${REPO_ROOT}/neural/experiments/Rascal_III_runner2778/train_gpt.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "ERROR: missing runner script at ${TRAIN_SCRIPT}" >&2
  exit 1
fi

echo "rascal_iii_runner2778_ckpt_export_script:${TRAIN_SCRIPT}"
echo "rascal_iii_runner2778_ckpt_export_profile init=${INIT_MODEL_PATH} seed=${SEED} nproc=${NPROC_PER_NODE} iterations=${ITERATIONS} warmup_steps=${WARMUP_STEPS} ttt_epochs=${TTT_EPOCHS} quant_bits=${QUANT_ATTN_BITS}/${QUANT_MLP_BITS}/${QUANT_AUX_BITS}/${QUANT_EMBED_BITS}/${QUANT_OTHER_BITS} quant_roundtrip_eval=${QUANT_ROUNDTRIP_EVAL} skip_final_eval=${SKIP_FINAL_EVAL}"

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@"
