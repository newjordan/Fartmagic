#!/bin/bash
set -euo pipefail
# ================================================================
# BW XII Quant Fix Round 2 — Per-Firing Identity + Throttles
#
# Root cause: shared crawler fires 9x with ZERO per-firing signal.
# FLOW is gated on crawler_loops>1 — disabled in helix (loops=1).
# The crawler has no routing signal, forcing all context into weights.
# Quantization destroys these implicit context encodings.
#
# This round tests:
#   R1: HELIX_FIRE_EMBED=1 (per-firing learned embedding — main hypothesis)
#   R2: + HELIX_INJECT_CAP=1.0 (cap cross-injection magnitude growth)
#   R3: + HELIX_MERGE_CAP=0.7 (cap merge gate to preserve flat baseline)
#   R4: all three combined
#
# Control (R0) reuses T0_wd012 checkpoint from round 1 if available.
#
# Estimated: ~5 hours on 1×H100.
#
# Usage:
#   SEED=444 bash crawler/2026-04-04_BWXII_QuantFix/run_quantfix_round2_1gpu.sh
#
# Can chain after round 1:
#   SEED=444 bash crawler/2026-04-04_BWXII_QuantFix/run_quantfix_1gpu.sh && \
#   SEED=444 bash crawler/2026-04-04_BWXII_QuantFix/run_quantfix_round2_1gpu.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results_round2"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/round2_summary_s${SEED}_${TS}.tsv"

if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN=(torchrun)
else
    TORCHRUN=(python3 -m torch.distributed.run)
fi

# ----------------------------------------------------------------
# BW XII SplitHead base config at gate scale (2000 steps, 1×GPU)
# ----------------------------------------------------------------
BASE_ENV=(
    SEED="${SEED}"
    ITERATIONS=2000
    MAX_WALLCLOCK_SECONDS=3600
    TRAIN_BATCH_TOKENS=786432
    WARMDOWN_ITERS=2000
    COMPLEMENT_ALPHA=0
    XSA_LAST_N=11
    BIGRAM_VOCAB_SIZE=2048
    ROPE_DIMS=16
    SWA_EVERY=50
    MTP_NUM_HEADS=0
    LATE_QAT_THRESHOLD=0
    MATRIX_LR=0.03
    EMBED_LR=0.035
    TORCHDYNAMO_OPTIMIZE_DDP=0
    COMPILE_FULLGRAPH=1
    NGRAM_EVAL_ORDER=0
    MODEL_DIM=512
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=9
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=1
    CRAWLER_MLP_MULT=6.0
    INST_DIM=32
    DELTA_NET_HEADS=0
    SKIP_EMA=1
    QK_GAIN_INIT=4.0
    GPTQ_CAL_SAMPLES=128
    GPTQ_CAL_SEQ_LEN=2048
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_MLP_CHOKE_SHAPE=flat
    CRAWLER_MLP_CHOKE_GROUPS=8
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_LOOP_SMEAR=0
    CRAWLER_TAP_DIM=0
    CRAWLER_TAP_LOOP_SPECIFIC=1
    CRAWLER_TAP_LAYERS=all
    ANCHOR_DIM=0
    FLAT_WEIGHT_SHARE=0
    HELIX=1
    HELIX_DIM=192
    HELIX_STRIDE=1
    CRAWLER_CROSS_HEADS=4
    MUON_WD=0.12
    SMART_SKIP=0
    NPROC_PER_NODE=1
)

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
extract() {
    grep -oP "$1" "$2" 2>/dev/null | tail -1 || echo "?"
}

calc_gap() {
    local raw="$1" int6="$2"
    python3 -c "
r, i = '$raw', '$int6'
if r != '?' and i != '?':
    print(f'{float(i)-float(r):.4f}')
else:
    print('?')
" 2>/dev/null || echo "?"
}

# ----------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------
echo "[preflight] brotli..."
python3 -c "import brotli; print('  brotli OK')" 2>/dev/null \
    || { echo "  installing..."; pip install brotli -q; }

echo "[preflight] flash_attn..."
python3 - <<'PY'
try:
    import flash_attn_interface
    print("  FA3 (hopper) OK")
except Exception:
    try:
        import flash_attn
        print(f"  flash-attn v{flash_attn.__version__}")
    except Exception:
        raise SystemExit("  ERROR: flash-attn not importable")
PY

echo "[preflight] dataset..."
python3 - <<'PY'
import glob, os
tok = "./data/tokenizers/fineweb_1024_bpe.model"
assert os.path.isfile(tok), f"missing: {tok}"
shards = glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
assert len(shards) >= 8, f"need >=8 shards, found {len(shards)}"
print(f"  tokenizer OK, train shards={len(shards)}")
PY

# ----------------------------------------------------------------
# Summary header
# ----------------------------------------------------------------
{
    echo -e "phase\tarm\tfire_embed\tinject_cap\tmerge_cap\traw_bpb\tint6_sw_bpb\tquant_gap\tbytes\tquant_method\tlog"
} > "${SUMMARY}"

echo ""
echo "================================================================"
echo "  BW XII QUANT FIX ROUND 2 — Per-Firing Identity + Throttles"
echo "  seed=${SEED}  $(date)"
echo ""
echo "  Root cause: shared crawler has ZERO per-firing signal."
echo "  FLOW gated on loops>1 — disabled in helix (loops=1)."
echo ""
echo "  R1: fire_embed (main hypothesis)"
echo "  R2: fire_embed + inject_cap=1.0"
echo "  R3: fire_embed + merge_cap=0.7"
echo "  R4: fire_embed + inject_cap=1.0 + merge_cap=0.7"
echo "  summary: ${SUMMARY}"
echo "================================================================"
echo ""

# ================================================================
# PHASE 1 — Training arms
# ================================================================
declare -a TRAIN_NAMES=(  "R1_fire_embed"  "R2_embed_inject"  "R3_embed_merge"  "R4_full_fix"  )
declare -a TRAIN_FIRE=(   "1"              "1"                "1"               "1"            )
declare -a TRAIN_INJECT=( "0"              "1.0"              "0"               "1.0"          )
declare -a TRAIN_MERGE=(  "0"              "0"                "0.7"             "0.7"          )

for i in "${!TRAIN_NAMES[@]}"; do
    arm="${TRAIN_NAMES[$i]}"
    fe="${TRAIN_FIRE[$i]}"
    ic="${TRAIN_INJECT[$i]}"
    mc="${TRAIN_MERGE[$i]}"
    log="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.log"
    ckpt="${RESULTS_DIR}/${arm}_s${SEED}_${TS}.final_model.pt"

    echo ""
    echo "================================================================"
    echo "  PHASE 1 — ${arm}: FIRE_EMBED=${fe} INJECT_CAP=${ic} MERGE_CAP=${mc}"
    echo "  log: ${log}"
    echo "  started: $(date)"
    echo "================================================================"

    env "${BASE_ENV[@]}" \
        HELIX_FIRE_EMBED="${fe}" \
        HELIX_INJECT_CAP="${ic}" \
        HELIX_MERGE_CAP="${mc}" \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        CRAWLER_QUANT_INT8=0 \
        SKIP_TRAIN=0 \
        INIT_MODEL_PATH="" \
        "${TORCHRUN[@]}" --standalone --nproc_per_node=1 "${TRAIN_PY}" \
        2>&1 | tee "${log}"

    if [[ -f "${REPO_ROOT}/final_model.pt" ]]; then
        cp -f "${REPO_ROOT}/final_model.pt" "${ckpt}"
        echo "  checkpoint saved: ${ckpt}"
    else
        echo "  WARNING: no final_model.pt found!"
    fi

    raw=$(extract 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    int6=$(extract 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
    bytes=$(extract 'Total submission size int6\+(?:brotli|zlib): \K[0-9]+' "${log}")
    gap=$(calc_gap "${raw}" "${int6}")

    echo -e "train\t${arm}\t${fe}\t${ic}\t${mc}\t${raw}\t${int6}\t${gap}\t${bytes}\tnaive_int6\t${log}" >> "${SUMMARY}"
    echo "  >>> ${arm}: raw=${raw} int6_sw=${int6} gap=${gap} bytes=${bytes}"
done

echo ""
echo "================================================================"
echo "  PHASE 1 COMPLETE — $(date)"
echo "================================================================"
echo ""

# ================================================================
# PHASE 2 — GPTQ sweep on each checkpoint (SKIP_TRAIN=1)
# ================================================================
declare -a QUANT_NAMES=(  "Q1_gptq_loop"      "Q2_gptq_loop_int8"  )
declare -a QUANT_SKIP=(   "0"                  "0"                  )
declare -a QUANT_LOOP=(   "1"                  "1"                  )
declare -a QUANT_INT8=(   "0"                  "1"                  )
declare -a QUANT_DESC=(   "loop-aware GPTQ"    "loop-aware+int8"    )

for i in "${!TRAIN_NAMES[@]}"; do
    train_arm="${TRAIN_NAMES[$i]}"
    fe="${TRAIN_FIRE[$i]}"
    ic="${TRAIN_INJECT[$i]}"
    mc="${TRAIN_MERGE[$i]}"
    ckpt="${RESULTS_DIR}/${train_arm}_s${SEED}_${TS}.final_model.pt"

    if [[ ! -f "${ckpt}" ]]; then
        echo "  SKIP Phase 2 for ${train_arm} — no checkpoint found"
        continue
    fi

    for j in "${!QUANT_NAMES[@]}"; do
        qarm="${QUANT_NAMES[$j]}"
        full_arm="${train_arm}_${qarm}"
        log="${RESULTS_DIR}/${full_arm}_s${SEED}_${TS}.log"

        echo ""
        echo "----------------------------------------------------------------"
        echo "  PHASE 2 — ${full_arm}: ${QUANT_DESC[$j]}"
        echo "  checkpoint: ${ckpt}"
        echo "  started: $(date)"
        echo "----------------------------------------------------------------"

        env "${BASE_ENV[@]}" \
            HELIX_FIRE_EMBED="${fe}" \
            HELIX_INJECT_CAP="${ic}" \
            HELIX_MERGE_CAP="${mc}" \
            SKIP_GPTQ="${QUANT_SKIP[$j]}" \
            LOOP_AWARE_GPTQ="${QUANT_LOOP[$j]}" \
            CRAWLER_QUANT_INT8="${QUANT_INT8[$j]}" \
            SKIP_TRAIN=1 \
            INIT_MODEL_PATH="${ckpt}" \
            "${TORCHRUN[@]}" --standalone --nproc_per_node=1 "${TRAIN_PY}" \
            2>&1 | tee "${log}"

        raw=$(extract 'DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
        int6=$(extract 'final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${log}")
        bytes=$(extract 'Total submission size int6\+(?:brotli|zlib): \K[0-9]+' "${log}")
        gap=$(calc_gap "${raw}" "${int6}")

        echo -e "quant\t${full_arm}\t${fe}\t${ic}\t${mc}\t${raw}\t${int6}\t${gap}\t${bytes}\t${QUANT_DESC[$j]}\t${log}" >> "${SUMMARY}"
        echo "  >>> ${full_arm}: raw=${raw} int6_sw=${int6} gap=${gap} bytes=${bytes}"
    done
done

# ----------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  BW XII QUANT FIX ROUND 2 — ALL COMPLETE"
echo "  $(date)"
echo "  summary: ${SUMMARY}"
echo ""
echo "  Phase 1: 4 training arms (per-firing identity + throttles)"
echo "    R1: fire_embed only (MAIN HYPOTHESIS)"
echo "    R2: fire_embed + inject_cap=1.0"
echo "    R3: fire_embed + merge_cap=0.7"
echo "    R4: fire_embed + inject_cap=1.0 + merge_cap=0.7"
echo ""
echo "  Phase 2: 2 GPTQ strategies × 4 checkpoints"
echo ""
echo "  Compare quant_gap to Round 1 T0 control (gap=0.023 naive int6)."
echo "  If fire_embed closes gap significantly → FLOW was the missing piece."
echo "================================================================"
echo ""
column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
