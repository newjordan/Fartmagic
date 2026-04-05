#!/bin/bash
set -euo pipefail
# ================================================================
# Smokestack — 3-arm cadence gate (2000 steps)
#
# Tests CRAWLER_LOOPS={3,2,1} on BWX 9F stable base.
# ONE variable: loop cadence. Everything else locked.
#
# Usage:
#   SEED=444 NPROC_PER_NODE=4 bash experiments/smokestack/gate.sh
#
# Optional:
#   NPROC_PER_NODE=1   # 1-GPU gate (~$0.50)
#   NPROC_PER_NODE=4   # 4-GPU gate (faster, ~$1)
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
RUN_TAG="SMKSTK"
TS="$(date +%Y%m%d_%H%M%S)"

GATE_ITERATIONS="${GATE_ITERATIONS:-2000}"
GATE_MAX_WALLCLOCK_SECONDS="${GATE_MAX_WALLCLOCK_SECONDS:-3600}"
GATE_TRAIN_BATCH_TOKENS="${GATE_TRAIN_BATCH_TOKENS:-786432}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary_s${SEED}_${TS}.tsv"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

# ----------------------------------------------------------------
# BWX 9F locked base environment (do NOT change these)
# ----------------------------------------------------------------
BASE_ENV=(
    SEED="${SEED}"
    ITERATIONS="${GATE_ITERATIONS}"
    MAX_WALLCLOCK_SECONDS="${GATE_MAX_WALLCLOCK_SECONDS}"
    TRAIN_BATCH_TOKENS="${GATE_TRAIN_BATCH_TOKENS}"
    WARMDOWN_ITERS=2000
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    ROPE_DIMS=16
    SWA_EVERY=50
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR=0.03
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_FULLGRAPH=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=9
    NUM_CRAWLER_LAYERS=1
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    CRAWLER_QUANT_INT8=0
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    SKIP_GPTQ=1
    LOOP_AWARE_GPTQ=0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_MLP_CHOKE_SHAPE=flat
    CRAWLER_MLP_CHOKE_GROUPS=8
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    NPROC_PER_NODE="${NPROC}"
)

# ----------------------------------------------------------------
# Arm definitions: ONLY CRAWLER_LOOPS and ROPE_SCALES change
# ----------------------------------------------------------------
declare -a ARM_NAMES=("SMKSTK-00" "SMKSTK-01" "SMKSTK-02")
declare -a ARM_DESCS=(
    "control: 9F, 1 crawler, loops=3, rope=(9,1,1)"
    "cadence down: loops=2"
    "cadence minimal: loops=1"
)
declare -a ARM_LOOPS=(3 2 1)
declare -a ARM_ROPE_SCALES=("9,1,1" "9,1" "9")

declare -A CONTROL_INT6=()

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
extract_metric() {
    local pattern="$1"
    local logfile="$2"
    grep -oP "${pattern}" "${logfile}" | tail -1 || true
}

is_numeric() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

calc_delta() {
    local control="$1"
    local value="$2"
    if ! is_numeric "${control}" || ! is_numeric "${value}"; then
        echo "?"
        return
    fi
    python3 - "${control}" "${value}" <<'PY'
import sys
c = float(sys.argv[1])
v = float(sys.argv[2])
d = v - c
sign = "+" if d >= 0 else ""
print(f"{sign}{d:.6f}")
PY
}

# ----------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || { echo "  ERROR: zstandard missing (pip install zstandard)"; exit 1; }

echo "[preflight] checking flash_attn..."
python3 - <<'PY'
try:
    import flash_attn_interface
    print("  FA3 (hopper) OK")
except Exception:
    try:
        import flash_attn
        v = flash_attn.__version__
        if str(v).startswith("3"):
            print(f"  FA3 v{v} OK")
        else:
            print(f"  WARNING: flash-attn v{v} detected (want v3)")
    except Exception:
        raise SystemExit("  ERROR: flash-attn not importable")
PY

echo "[preflight] checking dataset + tokenizer..."
python3 - <<'PY'
import glob, os
tok = "./data/tokenizers/fineweb_1024_bpe.model"
assert os.path.isfile(tok), f"missing tokenizer: {tok}"
shards = glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
assert len(shards) >= 8, f"need >=8 train shards, found {len(shards)}"
print(f"  tokenizer OK, train shards={len(shards)}")
PY

# ----------------------------------------------------------------
# Summary header
# ----------------------------------------------------------------
{
    echo -e "arm\tdesc\tmodel_params\traw_bpb\tint6_sw_bpb\tstep_ms\tbytes\tdelta_vs_control\tlog"
} > "${SUMMARY}"

echo ""
echo "============================================"
echo "  Smokestack — Cadence Gate"
echo "  seed=${SEED} GPUs=${NPROC} iters=${GATE_ITERATIONS}"
echo "  Arms: loops=3 (control), loops=2, loops=1"
echo "  summary: ${SUMMARY}"
echo "============================================"
echo ""

# ----------------------------------------------------------------
# Run each arm sequentially
# ----------------------------------------------------------------
for i in "${!ARM_NAMES[@]}"; do
    arm="${ARM_NAMES[$i]}"
    desc="${ARM_DESCS[$i]}"
    loops="${ARM_LOOPS[$i]}"
    rope="${ARM_ROPE_SCALES[$i]}"
    log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ${arm}: ${desc}"
    echo "  CRAWLER_LOOPS=${loops}  CRAWLER_LOOP_ROPE_SCALES=${rope}"
    echo "  log: ${log}"
    echo "----------------------------------------------------------------"

    env "${BASE_ENV[@]}" \
        CRAWLER_LOOPS="${loops}" \
        CRAWLER_LOOP_ROPE_SCALES="${rope}" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" "${TRAIN_PY}" \
        2>&1 | tee "${log}"

    # Save checkpoint
    if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
        cp -f "${REPO_ROOT}/final_model.pt" "${RESULTS_DIR}/${arm}_s${SEED}_${TS}.final_model.pt"
    fi

    # Extract metrics
    params=$(extract_metric 'model_params:\K[0-9]+' "${log}")
    raw=$(extract_metric 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    int6=$(extract_metric 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    step_ms=$(extract_metric 'step_avg:\K[0-9.]+' "${log}")
    bytes=$(extract_metric 'Total submission size int6\+(?:zstd|zlib): \K[0-9]+' "${log}")

    [[ -z "${params}" ]] && params="?"
    [[ -z "${raw}" ]] && raw="?"
    [[ -z "${int6}" ]] && int6="?"
    [[ -z "${step_ms}" ]] && step_ms="?"
    [[ -z "${bytes}" ]] && bytes="?"

    if [[ "${i}" -eq 0 ]]; then
        CONTROL_INT6["gate"]="${int6}"
    fi
    delta=$(calc_delta "${CONTROL_INT6[gate]:-}" "${int6}")

    echo -e "${arm}\t${desc}\t${params}\t${raw}\t${int6}\t${step_ms}\t${bytes}\t${delta}\t${log}" >> "${SUMMARY}"
    echo "  ${arm}: raw=${raw} int6_sw=${int6} step_ms=${step_ms} bytes=${bytes} delta_vs_ctrl=${delta}"
done

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
cat <<TXT

================================================================
Smokestack cadence gate complete.
summary: ${SUMMARY}

Arms tested:
  SMKSTK-00: control (loops=3)
  SMKSTK-01: loops=2  (BW17 RAPID signal: -0.054)
  SMKSTK-02: loops=1  (aggressive probe)

Pass criteria: delta_vs_control <= -0.003 int6_sw_bpb
================================================================
TXT

column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
