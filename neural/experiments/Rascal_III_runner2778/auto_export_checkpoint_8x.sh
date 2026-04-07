#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

if [ -n "${INIT_MODEL_PATH:-}" ]; then
  echo "auto_ckpt: using explicit INIT_MODEL_PATH=${INIT_MODEL_PATH}"
  exec bash "${SCRIPT_DIR}/export_checkpoint_8x.sh" "$@"
fi

# Colon-separated search roots; default to pod workspace then repo root.
CKPT_SEARCH_ROOTS="${CKPT_SEARCH_ROOTS:-/workspace/parameter-golf:${REPO_ROOT}}"
export CKPT_SEARCH_ROOTS

CKPT_PATH="$(
python3 - <<'PY'
import glob
import os
import torch

required = {"tok_emb.weight", "skip_weights", "bigram.ngram_gate", "qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank"}
prefixes = ("module.", "_orig_mod.", "model.", "base_model.")

def unwrap(obj):
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    return None

def normalize_keys(state):
    out = {}
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
        out[nk] = v
    return out

roots = [r for r in os.environ.get("CKPT_SEARCH_ROOTS", "").split(":") if r]
candidates = []
seen = set()
for root in roots:
    if not os.path.isdir(root):
        continue
    pattern = os.path.join(root, "**", "*.pt")
    for p in glob.iglob(pattern, recursive=True):
        if p in seen:
            continue
        seen.add(p)
        try:
            obj = torch.load(p, map_location="cpu")
        except Exception:
            continue
        state = unwrap(obj)
        if state is None:
            continue
        nstate = normalize_keys(state)
        if required.issubset(nstate.keys()):
            candidates.append((os.path.getmtime(p), p))

candidates.sort(reverse=True)
if candidates:
    print(candidates[0][1])
PY
)"

if [ -z "${CKPT_PATH}" ]; then
  echo "ERROR: no compatible Rascal_III_runner2778 checkpoint found." >&2
  echo "Searched roots: ${CKPT_SEARCH_ROOTS}" >&2
  echo "You can set INIT_MODEL_PATH explicitly or widen CKPT_SEARCH_ROOTS." >&2
  exit 2
fi

echo "auto_ckpt: selected checkpoint=${CKPT_PATH}"
export INIT_MODEL_PATH="${CKPT_PATH}"
exec bash "${SCRIPT_DIR}/export_checkpoint_8x.sh" "$@"
