#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
pip install brotli zstandard 2>/dev/null || true
python3 -c "import brotli" || { echo "FATAL: brotli not installed"; exit 1; }

export INIT_MODEL_PATH="${INIT_MODEL_PATH:-$ROOT/final_model_12L_brotli_mixedint_s444.pt}"
if [[ ! -f "$INIT_MODEL_PATH" ]]; then
    echo "FATAL: checkpoint not found at $INIT_MODEL_PATH"
    echo "Run: cp final_model.pt final_model_12L_brotli_mixedint_s444.pt"
    exit 1
fi
echo "Using checkpoint: $INIT_MODEL_PATH"

export SEED="${SEED:-444}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export ITERATIONS=0
export WARMUP_STEPS=0
export MAX_WALLCLOCK_SECONDS=0
export COMPRESSOR="${COMPRESSOR:-brotli}"
export LOADER_MODE=coprime
export COPRIME_SHARDS_PER_BATCH=1
export COPRIME_SHARD_HOLD_STEPS=64
export TRIGRAM=0
export NGRAM_EVAL_ORDER=0
export NUM_LAYERS=12

TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"
RESULTS_DIR="$SCRIPT_DIR/sweep_results"
mkdir -p "$RESULTS_DIR"

run_arm() {
    local arm_name="$1"
    local skip_gptq="$2"
    local attn_bits="$3"
    local mlp_bits="$4"
    local embed_bits="$5"

    echo ""
    echo "============================================"
    echo "  ARM: $arm_name"
    echo "  SKIP_GPTQ=$skip_gptq ATTN=$attn_bits MLP=$mlp_bits EMBED=$embed_bits"
    echo "============================================"

    export SKIP_GPTQ="$skip_gptq"
    export QUANT_ATTN_BITS="$attn_bits"
    export QUANT_MLP_BITS="$mlp_bits"
    export QUANT_EMBED_BITS="$embed_bits"
    export SKIP_FINAL_EVAL=0
    export POST_EMA_DIAGNOSTIC=1
    export QUANT_ARTIFACT_PATH="final_model.${arm_name}.ptz"

    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "$TRAIN_SCRIPT" \
        2>&1 | tee "$RESULTS_DIR/${arm_name}.log"

    local sw_bpb=$(grep "final_sliding_window_exact" "$RESULTS_DIR/${arm_name}.log" | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    local rt_bpb=$(grep "final_quant_roundtrip_exact" "$RESULTS_DIR/${arm_name}.log" | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    local size=$(grep "Total submission size" "$RESULTS_DIR/${arm_name}.log" | grep -oP ': \K[0-9]+' || echo "N/A")
    echo "$arm_name	$skip_gptq	$attn_bits	$mlp_bits	$embed_bits	$sw_bpb	$rt_bpb	$size" >> "$RESULTS_DIR/summary.tsv"
}

echo "arm	skip_gptq	attn_bits	mlp_bits	embed_bits	sw_bpb	rt_bpb	total_size" > "$RESULTS_DIR/summary.tsv"

# Arm A: naive all-int6 + brotli (baseline quant quality)
run_arm "A_naive_int6" 1 6 6 6

# Arm B: naive mixed-int (what we just ran — control)
run_arm "B_naive_mix" 1 5 6 8

# Arm C: GPTQ + mixed-int (calibrated int5 should close gap)
run_arm "C_gptq_mix" 0 5 6 8

# Arm D: GPTQ + all-int6 (best quant quality, slightly bigger)
run_arm "D_gptq_int6" 0 6 6 6

# Arm E: naive int6 attn, int6 mlp, int8 embed (relax attn to int6, keep embed at int8)
run_arm "E_naive_relax" 1 6 6 8

echo ""
echo "============================================"
echo "  SWEEP COMPLETE — 12-Layer Results:"
echo "  Post-EMA BPB: 1.1292 (target to recover)"
echo "============================================"
cat "$RESULTS_DIR/summary.tsv" | column -t -s$'\t'
echo ""
echo "Results saved to: $RESULTS_DIR/summary.tsv"
