#!/usr/bin/env bash
set -euo pipefail

# Base model tuning sweep: 7 arms, 2k steps each, 4 GPUs, seed 444
# Control = first ablation run (Lucky IV unmodified — use muoneq_r/qk4 ctrl proxy)
# Each arm changes ONE env var vs Lucky IV defaults

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../ && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/flash-attention/hopper${PYTHONPATH:+:${PYTHONPATH}}"
export SEED=444
export ITERATIONS=2000
export WARMDOWN_ITERS=0
export SKIP_FINAL_EVAL=1
export POST_EMA_DIAGNOSTIC=1
export COPRIME_MAX_LOADED_SHARDS=1
export COPRIME_SHARDS_PER_BATCH=1

pip install brotli 2>/dev/null || true

NPROC="${NPROC_PER_NODE:-4}"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
TRAIN_SCRIPT="${ROOT}/neural/experiments/Lucky_IV/train_gpt.py"
LOG_DIR="${SCRIPT_DIR}/base_sweep_logs"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "BASE MODEL SWEEP — $(date)"
echo "Arms: wd09, lr02, embed03, qk5, ns7, softcap50, clip1"
echo "Steps: 2000 | GPUs: ${NPROC} | Seed: ${SEED}"
echo "============================================"

run_arm() {
    local name="$1"
    shift
    echo ""
    echo ">>> ${name}"
    echo ">>> Env: $@"
    echo ">>> Started: $(date)"
    env "$@" python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
        "${TRAIN_SCRIPT}" 2>&1 | tee "${LOG_DIR}/${name}.txt"
    echo ">>> ${name} DONE: $(date)"
}

# --- ARM 1: Weight decay 0.09 (from 0.04) ---
run_arm "wd09" MUON_WD=0.09

# --- ARM 2: Matrix LR 0.02 (from 0.025) ---
run_arm "lr02" MATRIX_LR=0.02

# --- ARM 3: Embed LR 0.03 (from 0.035) ---
run_arm "embed03" TIED_EMBED_LR=0.03

# --- ARM 4: QK Gain 5.0 (from 1.5) ---
run_arm "qk5" QK_GAIN_INIT=5.0

# --- ARM 5: Newton-Schulz 7 steps (from 5) ---
run_arm "ns7" MUON_BACKEND_STEPS=7

# --- ARM 6: Logit softcap 50 (from 30) ---
run_arm "softcap50" LOGIT_SOFTCAP=50.0

# --- ARM 7: Grad clip 1.0 (from 0.3) ---
run_arm "clip1" GRAD_CLIP_NORM=1.0

echo ""
echo "============================================"
echo "ALL ARMS COMPLETE — $(date)"
echo "Logs in: ${LOG_DIR}/"
ls -la "${LOG_DIR}/"
echo "============================================"
