#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIR="${SCRIPT_DIR}"
while [ "${DIR}" != "/" ]; do [ -d "${DIR}/data/tokenizers" ] && break; DIR="$(dirname "${DIR}")"; done
if [ "${DIR}" = "/" ]; then echo "ERROR: could not find data/tokenizers/" >&2; exit 1; fi
export REPO_ROOT="${DIR}"
cd "${REPO_ROOT}"

export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CKPT_SEARCH_ROOTS="${CKPT_SEARCH_ROOTS:-/workspace/parameter-golf:${REPO_ROOT}}"
export QUANT_ROUNDTRIP_EVAL="${QUANT_ROUNDTRIP_EVAL:-1}"
export SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
export SWEEP_BASELINE_NAME="${SWEEP_BASELINE_NAME:-q0_base}"
export SWEEP_MAX_BPB_DELTA="${SWEEP_MAX_BPB_DELTA:-0.015}"
export SWEEP_SIZE_CAP_BYTES="${SWEEP_SIZE_CAP_BYTES:-16777216}"

# Never run final sliding/ngram eval inside this sweep unless explicitly overridden.
if [ "${SKIP_FINAL_EVAL}" != "1" ]; then
  echo "ERROR: sweep is designed for export-only; set SKIP_FINAL_EVAL=1." >&2
  exit 2
fi

if [ -z "${INIT_MODEL_PATH:-}" ] || [ ! -f "${INIT_MODEL_PATH}" ]; then
  # Direct CPU-only finder. Never launches torchrun.
  INIT_MODEL_PATH="$(
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
    for p in glob.iglob(os.path.join(root, "**", "*.pt"), recursive=True):
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
fi

if [ -z "${INIT_MODEL_PATH:-}" ] || [ ! -f "${INIT_MODEL_PATH}" ]; then
  echo "ERROR: no compatible runner2778 checkpoint found (searched ${CKPT_SEARCH_ROOTS})." >&2
  exit 3
fi
export INIT_MODEL_PATH

ts="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="neural/experiments/Rascal_III_runner2778/sweeps/${ts}_seed${SEED}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
RANKED="${OUT_DIR}/ranked.tsv"
BEST="${OUT_DIR}/best.txt"

echo -e "name\tattn\tmlp\taux\tembed\tother\tartifact_bytes\ttotal_bytes\troundtrip_bpb\tstatus\tlog_path" > "${SUMMARY}"

# Small, bounded sweep to limit burn.
# Format: name attn mlp aux embed other
CONFIGS=(
  "q0_base 5 6 6 8 8"
  "q1_mlp5 5 5 6 8 8"
  "q2_attn4 4 6 6 8 8"
  "q3_attn4_aux5 4 6 5 8 8"
  "q4_attn4_mlp5_aux6 4 5 6 8 8"
)

echo "sweep:start checkpoint=${INIT_MODEL_PATH} out_dir=${OUT_DIR} configs=${#CONFIGS[@]} baseline=${SWEEP_BASELINE_NAME} max_bpb_delta=${SWEEP_MAX_BPB_DELTA} size_cap=${SWEEP_SIZE_CAP_BYTES}"

for cfg in "${CONFIGS[@]}"; do
  read -r name attn mlp aux embed other <<< "${cfg}"
  run_id="rascal_iii_runner2778_qsweep_${name}_s${SEED}_${ts}"
  log_path="logs/${run_id}.txt"

  echo "sweep:run name=${name} bits=${attn}/${mlp}/${aux}/${embed}/${other}"

  if INIT_MODEL_PATH="${INIT_MODEL_PATH}" \
     SEED="${SEED}" \
     NPROC_PER_NODE="${NPROC_PER_NODE}" \
     RUN_ID="${run_id}" \
     ITERATIONS=0 WARMUP_STEPS=0 WARMDOWN_ITERS=0 \
     SWA_ENABLED=0 POST_EMA_DIAGNOSTIC=0 \
     TTT_EPOCHS=0 TTT_LR=0.0 TTT_FREEZE_BLOCKS=0 \
     MAX_WALLCLOCK_SECONDS=0 \
     QUANT_ATTN_BITS="${attn}" \
     QUANT_MLP_BITS="${mlp}" \
     QUANT_AUX_BITS="${aux}" \
     QUANT_EMBED_BITS="${embed}" \
     QUANT_OTHER_BITS="${other}" \
     QUANT_ROUNDTRIP_EVAL="${QUANT_ROUNDTRIP_EVAL}" \
     SKIP_FINAL_EVAL=1 \
     bash "${SCRIPT_DIR}/export_checkpoint_8x.sh"
  then
    artifact_bytes="$(awk '/Serialized model mixed\+/{print $(NF-1)}' "${log_path}" | tail -1)"
    total_bytes="$(awk '/Total submission size/{print $(NF-1)}' "${log_path}" | tail -1)"
    roundtrip_bpb="$(awk -F'val_bpb:' '/final_quant_roundtrip_exact/{print $2}' "${log_path}" | awk '{print $1}' | tail -1)"
    artifact_bytes="${artifact_bytes:-}"
    total_bytes="${total_bytes:-}"
    roundtrip_bpb="${roundtrip_bpb:-}"
    echo -e "${name}\t${attn}\t${mlp}\t${aux}\t${embed}\t${other}\t${artifact_bytes}\t${total_bytes}\t${roundtrip_bpb}\tok\t${log_path}" >> "${SUMMARY}"
  else
    echo -e "${name}\t${attn}\t${mlp}\t${aux}\t${embed}\t${other}\t\t\t\tfail\t${log_path}" >> "${SUMMARY}"
  fi
done

python3 - <<'PY' "${SUMMARY}" "${RANKED}" "${BEST}" "${SWEEP_BASELINE_NAME}" "${SWEEP_MAX_BPB_DELTA}" "${SWEEP_SIZE_CAP_BYTES}"
import csv
import math
import sys

summary, ranked, best, baseline_name, max_delta_str, size_cap_str = sys.argv[1:7]
max_delta = float(max_delta_str)
size_cap = int(size_cap_str)
rows = []
with open(summary, "r", encoding="utf-8") as f:
    rd = csv.DictReader(f, delimiter="\t")
    for r in rd:
        if r["status"] != "ok":
            continue
        try:
            total = int(r["total_bytes"])
        except Exception:
            total = 10**18
        try:
            bpb = float(r["roundtrip_bpb"]) if r["roundtrip_bpb"] else math.inf
        except Exception:
            bpb = math.inf
        r["_total"] = total
        r["_bpb"] = bpb
        rows.append(r)

baseline = next((r for r in rows if r["name"] == baseline_name), None)
baseline_bpb = baseline["_bpb"] if baseline is not None else (min((r["_bpb"] for r in rows), default=math.inf))

for r in rows:
    r["_delta_bpb"] = r["_bpb"] - baseline_bpb
    r["_quality_ok"] = r["_delta_bpb"] <= max_delta
    r["_size_ok"] = r["_total"] <= size_cap
    if r["_quality_ok"] and r["_size_ok"]:
        r["_tier"] = 0
    elif r["_quality_ok"]:
        r["_tier"] = 1
    elif r["_size_ok"]:
        r["_tier"] = 2
    else:
        r["_tier"] = 3

def sort_key(r):
    # Tier 0: quality+size pass -> prefer smaller.
    if r["_tier"] == 0:
        return (0, r["_total"], r["_bpb"])
    # Tier 1: quality pass but oversize -> prefer closer to cap then better bpb.
    if r["_tier"] == 1:
        return (1, r["_total"], r["_bpb"])
    # Tier 2: size pass but quality fail -> prefer better quality first.
    if r["_tier"] == 2:
        return (2, r["_bpb"], r["_total"])
    # Tier 3: fail both -> prefer better quality first.
    return (3, r["_bpb"], r["_total"])

rows.sort(key=sort_key)

with open(ranked, "w", encoding="utf-8", newline="") as f:
    wr = csv.writer(f, delimiter="\t")
    wr.writerow([
        "rank","name","attn","mlp","aux","embed","other",
        "total_bytes","roundtrip_bpb","delta_bpb","quality_ok","size_ok","tier","log_path"
    ])
    for i, r in enumerate(rows, start=1):
        wr.writerow([
            i, r["name"], r["attn"], r["mlp"], r["aux"], r["embed"], r["other"],
            r["total_bytes"], r["roundtrip_bpb"],
            f"{r['_delta_bpb']:.8f}", int(r["_quality_ok"]), int(r["_size_ok"]), r["_tier"], r["log_path"]
        ])

with open(best, "w", encoding="utf-8") as f:
    if not rows:
        f.write("no_successful_runs\n")
    else:
        r = rows[0]
        f.write(f"baseline_name={baseline_name}\n")
        f.write(f"baseline_bpb={baseline_bpb:.8f}\n")
        f.write(f"max_bpb_delta={max_delta:.8f}\n")
        f.write(f"size_cap_bytes={size_cap}\n")
        f.write(f"best_name={r['name']}\n")
        f.write(f"bits={r['attn']}/{r['mlp']}/{r['aux']}/{r['embed']}/{r['other']}\n")
        f.write(f"total_bytes={r['total_bytes']}\n")
        f.write(f"roundtrip_bpb={r['roundtrip_bpb']}\n")
        f.write(f"delta_bpb={r['_delta_bpb']:.8f}\n")
        f.write(f"quality_ok={int(r['_quality_ok'])}\n")
        f.write(f"size_ok={int(r['_size_ok'])}\n")
        f.write(f"tier={r['_tier']}\n")
        f.write(f"log_path={r['log_path']}\n")
PY

echo "sweep:done summary=${SUMMARY} ranked=${RANKED} best=${BEST}"
