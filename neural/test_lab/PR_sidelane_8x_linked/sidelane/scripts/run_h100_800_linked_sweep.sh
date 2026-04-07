#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="${REPO_ROOT}"
elif git -C "${ROOT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "${ROOT_DIR}" rev-parse --show-toplevel)"
else
  REPO_ROOT="$(cd -- "${ROOT_DIR}/../../../../.." && pwd)"
fi
PG_LAB_ROOT="${PG_LAB_ROOT:-${REPO_ROOT}/neural}"

if [[ -z "${TRAIN_PY:-}" ]]; then
  for candidate in \
    "${PG_LAB_ROOT}/experiments/rascal_hunt_2k/train_gpt.py" \
    "${PG_LAB_ROOT}/experiments/Rascal_III/train_gpt.py" \
    "${REPO_ROOT}/train_gpt.py"; do
    if [[ -f "${candidate}" ]]; then
      TRAIN_PY="${candidate}"
      break
    fi
  done
fi

if [[ -z "${TOKENIZER_PATH:-}" ]]; then
  for candidate in \
    "${PG_LAB_ROOT}/data/tokenizers/fineweb_1024_bpe.model" \
    "${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model" \
    "${REPO_ROOT}/data/fineweb_1024_bpe.model"; do
    if [[ -f "${candidate}" ]]; then
      TOKENIZER_PATH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ -f "${ROOT_DIR}/data/fineweb_sp1024_smoke/fineweb_train_000000.bin" && -f "${ROOT_DIR}/data/fineweb_sp1024_smoke/fineweb_val_000000.bin" ]]; then
    DATA_PATH="${ROOT_DIR}/data/fineweb_sp1024_smoke"
  elif [[ -f "${REPO_ROOT}/data/fineweb_sp1024_smoke/fineweb_train_000000.bin" && -f "${REPO_ROOT}/data/fineweb_sp1024_smoke/fineweb_val_000000.bin" ]]; then
    DATA_PATH="${REPO_ROOT}/data/fineweb_sp1024_smoke"
  elif [[ -f "/home/frosty40/parameter-golf-cross-repo/data/fineweb_sp1024_smoke/fineweb_train_000000.bin" && -f "/home/frosty40/parameter-golf-cross-repo/data/fineweb_sp1024_smoke/fineweb_val_000000.bin" ]]; then
    DATA_PATH="/home/frosty40/parameter-golf-cross-repo/data/fineweb_sp1024_smoke"
  else
    DATA_PATH="${PG_LAB_ROOT}/data/datasets/fineweb10B_sp1024"
  fi
fi

if [[ -z "${TRAIN_PY:-}" || ! -f "${TRAIN_PY}" ]]; then
  echo "TRAIN_PY not found. Set TRAIN_PY=/abs/path/to/train_gpt.py" >&2
  echo "Tried defaults under: ${PG_LAB_ROOT}/experiments and ${REPO_ROOT}" >&2
  exit 1
fi
if [[ -z "${TOKENIZER_PATH:-}" || ! -f "${TOKENIZER_PATH}" ]]; then
  echo "TOKENIZER_PATH not found. Set TOKENIZER_PATH=/abs/path/to/fineweb_1024_bpe.model" >&2
  exit 1
fi
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "DATA_PATH not found: ${DATA_PATH}" >&2
  echo "Set DATA_PATH=/abs/path/to/tokenized_dataset_dir" >&2
  exit 1
fi

if [[ "${SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  if ! python3 - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "CUDA not available in this shell. Run this on 1x H100 (or set SKIP_CUDA_CHECK=1)." >&2
    exit 1
  fi
fi

TORCHRUN=(python3 -m torch.distributed.run)

SEED="${SEED:-445}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RUNS_SUBDIR="${RUNS_SUBDIR:-h100_800_linked}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/runs/${RUNS_SUBDIR}/${TS}_s${SEED}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
LEADERBOARD="${OUT_DIR}/leaderboard.tsv"

# H100 800-step profile defaults.
export DATA_PATH TOKENIZER_PATH SEED
export ITERATIONS="${ITERATIONS:-800}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-2400}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-32768}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32768}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-256}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-256}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"

# Keep runs deterministic and cheap for comparison.
export COMPILE_ENABLED="${COMPILE_ENABLED:-0}"
export COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"
export TORCHDYNAMO_OPTIMIZE_DDP="${TORCHDYNAMO_OPTIMIZE_DDP:-0}"
export DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
export SKIP_GPTQ="${SKIP_GPTQ:-1}"
export SKIP_EMA="${SKIP_EMA:-1}"
export SWA_ENABLED="${SWA_ENABLED:-0}"

# Baseline backbone defaults (per-arm env files can override these).
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MODEL_DIM="${MODEL_DIM:-384}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export NUM_HEADS="${NUM_HEADS:-6}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-3}"
export MLP_MULT="${MLP_MULT:-3.0}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
export BIGRAM_DIM="${BIGRAM_DIM:-128}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export VE_ENABLED="${VE_ENABLED:-1}"
export VE_DIM="${VE_DIM:-128}"
export VE_LAYERS="${VE_LAYERS:-7,8}"
export DTG_ENABLED="${DTG_ENABLED:-0}"
export USE_CRAWLER="${USE_CRAWLER:-0}"
export NUM_FLAT_LAYERS="${NUM_FLAT_LAYERS:-9}"
export NUM_CRAWLER_LAYERS="${NUM_CRAWLER_LAYERS:-1}"
export CRAWLER_LOOPS="${CRAWLER_LOOPS:-3}"
export CRAWLER_MLP_MULT="${CRAWLER_MLP_MULT:-6.0}"
export INST_DIM="${INST_DIM:-32}"
export CRAWLER_LOOP_ROPE_SCALES="${CRAWLER_LOOP_ROPE_SCALES:-9,1,1}"
export CRAWLER_LOOP_SMEAR="${CRAWLER_LOOP_SMEAR:-0}"

# Linked control+candidate sequence across requested categories.
PAIRS=(
  "01_jepa|00_control|control"
  "01_jepa|13_jepa_proxy_alpha005_densehash|candidate"
  "02_text_diffusion|00_control|control"
  "02_text_diffusion|16_diff_proxy_denoise_complement_densehash|candidate"
  "03_hnet_tokenization|00_control|control"
  "03_hnet_tokenization|12_hnet_proxy_combo|candidate"
  "04_ssm_e2e_ttt_long_context|11_longctx_2k|control"
  "04_ssm_e2e_ttt_long_context|13_longctx_ttt_densehash|candidate"
  "05_adapters_random_linear_maps|00_control|control"
  "05_adapters_random_linear_maps|13_rlm_adapter_densehash|candidate"
)

printf "experiment\tarm\trole\tdesc\tstatus\tmodel_params\tdiag_bpb\tint6_bpb\tsw_bpb\ttotal_size_bytes\tstep_avg_ms\tlog\n" > "${SUMMARY}"

echo "============================================================"
echo "H100 linked sweep (800-step profile)"
echo "out_dir: ${OUT_DIR}"
echo "train_py: ${TRAIN_PY}"
echo "data_path: ${DATA_PATH}"
echo "tokenizer_path: ${TOKENIZER_PATH}"
echo "seed: ${SEED}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "runs_subdir: ${RUNS_SUBDIR}"
echo "iterations: ${ITERATIONS}"
echo "max_wallclock_seconds: ${MAX_WALLCLOCK_SECONDS}"
echo "============================================================"

for triple in "${PAIRS[@]}"; do
  IFS="|" read -r experiment arm_name arm_role <<< "${triple}"
  arm_file="${ROOT_DIR}/${experiment}/arms/${arm_name}.env"
  if [[ ! -f "${arm_file}" ]]; then
    echo "Missing arm file: ${arm_file}" >&2
    exit 1
  fi

  log_path="${OUT_DIR}/${experiment}__${arm_name}.log"
  echo
  echo "--- ${experiment} / ${arm_name} (${arm_role}) ---"

  arm_status="ok"
  if (
    set -a
    # shellcheck disable=SC1090
    source "${arm_file}"
    set +a
    export RUN_ID="h100800_${experiment}_${arm_name}_s${SEED}_${TS}"
    "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_PY}"
  ) 2>&1 | tee "${log_path}"; then
    arm_status="ok"
  else
    arm_status="fail"
    echo "arm_failed:${experiment}/${arm_name} (continuing linked sweep)" >&2
  fi

  arm_desc="$(grep -E '^ARM_DESC=' "${arm_file}" | head -1 | sed -E 's/^ARM_DESC="?//; s/"?$//' || true)"
  [[ -z "${arm_desc}" ]] && arm_desc="${arm_name}"

  model_params="$(grep -oP 'model_params:\K[0-9]+' "${log_path}" | tail -1 || true)"
  diag_bpb="$(grep -oP 'DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log_path}" | tail -1 || true)"
  int6_bpb="$(grep -oP 'final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log_path}" | tail -1 || true)"
  sw_bpb="$(grep -oP '(?:final_int8_zlib_roundtrip_exact|final_sliding_window(?:_s[0-9]+)?_exact) val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log_path}" | tail -1 || true)"
  total_size_bytes="$(grep -oP 'Total submission size int6\+\w+: \K[0-9]+' "${log_path}" | tail -1 || true)"
  step_avg_ms="$(grep -oP 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:[0-9.]+ train_time:[0-9]+ms step_avg:\K[0-9.]+' "${log_path}" | tail -1 || true)"

  [[ -z "${model_params}" ]] && model_params="-"
  [[ -z "${diag_bpb}" ]] && diag_bpb="-"
  [[ -z "${int6_bpb}" ]] && int6_bpb="-"
  [[ -z "${sw_bpb}" ]] && sw_bpb="-"
  [[ -z "${total_size_bytes}" ]] && total_size_bytes="-"
  [[ -z "${step_avg_ms}" ]] && step_avg_ms="-"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${experiment}" "${arm_name}" "${arm_role}" "${arm_desc}" "${arm_status}" "${model_params}" "${diag_bpb}" "${int6_bpb}" "${sw_bpb}" "${total_size_bytes}" "${step_avg_ms}" "${log_path}" \
    >> "${SUMMARY}"
done

python3 - "${SUMMARY}" "${LEADERBOARD}" <<'PY'
import csv
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
leaderboard_path = Path(sys.argv[2])

with summary_path.open() as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

def parse(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.inf

ok_rows = [r for r in rows if r.get("status") == "ok"]
for r in ok_rows:
    sw = parse(r.get("sw_bpb", ""))
    i6 = parse(r.get("int6_bpb", ""))
    r["_metric"] = sw if math.isfinite(sw) else i6

ok_rows = [r for r in ok_rows if math.isfinite(r.get("_metric", math.inf))]
ok_rows.sort(key=lambda r: r["_metric"])

with leaderboard_path.open("w") as f:
    f.write("rank\texperiment\tarm\trole\tmetric_bpb\tsw_bpb\tint6_bpb\ttotal_size_bytes\n")
    for idx, r in enumerate(ok_rows, start=1):
        f.write(
            f"{idx}\t{r['experiment']}\t{r['arm']}\t{r['role']}\t{r['_metric']:.8f}\t{r['sw_bpb']}\t{r['int6_bpb']}\t{r['total_size_bytes']}\n"
        )

print(f"summary={summary_path}")
print(f"leaderboard={leaderboard_path}")
if ok_rows:
    best = ok_rows[0]
    print(f"best={best['experiment']}/{best['arm']} metric_bpb={best['_metric']:.8f}")
else:
    print("best=unavailable")
PY

echo
echo "linked_sweep_complete"
echo "summary: ${SUMMARY}"
echo "leaderboard: ${LEADERBOARD}"
