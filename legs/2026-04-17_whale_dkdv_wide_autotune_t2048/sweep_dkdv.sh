#!/usr/bin/env bash
# Sweep dkdv config space via WHALE_BWD_KV_TMA_CONFIG env override.
# Each config runs bench_stable.py once at shape (2,2048,8,4,64) fused_delta_tma.
set -euo pipefail
cd /workspace/SOTA_FINAL/legs/2026-04-17_whale_dkdv_wide_autotune_t2048
source ./tracked_env.sh

TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="logs/sweep_${TS}.log"
mkdir -p logs evidence
BENCH="/workspace/SOTA_FINAL/legs/2026-04-16_whale_bwd_ablations/bench_stable.py"

CONFIGS=(
  "64,64,4,2"    "64,64,8,2"
  "64,128,4,2"   "64,128,8,2"
  "128,64,4,2"   "128,64,8,2"   "128,64,8,3"   "128,64,8,4"
  "128,128,4,2"  "128,128,8,2"  "128,128,4,3"  "128,128,8,3"
  "256,64,4,2"   "256,64,8,2"   "256,64,8,3"   "256,128,8,2"
  "64,256,4,2"   "64,256,8,2"
)

echo "[$TS] dkdv wide sweep — ${#CONFIGS[@]} configs, shape (2,2048,8,4,64)" | tee -a "$SWEEP_LOG"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tee -a "$SWEEP_LOG"

for cfg in "${CONFIGS[@]}"; do
  SAFE="${cfg//,/_}"
  RUN_TS="$(date +%Y%m%d_%H%M%S)"
  OUT="evidence/dkdv_${SAFE}_${RUN_TS}.json"
  LOG="logs/dkdv_${SAFE}_${RUN_TS}.log"
  echo "[$(date +%H:%M:%S)] --> ${cfg}  out=${OUT}" | tee -a "$SWEEP_LOG"
  # Fresh triton cache per config to avoid JIT artifact carryover
  rm -rf /root/.triton/cache 2>/dev/null || true
  WHALE_BWD_KV_TMA_CONFIG="${cfg}" \
    /venv/main/bin/python3 "$BENCH" \
      --label "dkdv_${SAFE}" \
      --shape "2,2048,8,4,64" \
      --backends whale_fast,fa3 \
      --rounds 5 --iters 100 \
      --out "$OUT" >>"$LOG" 2>&1 \
    && echo "  ok" | tee -a "$SWEEP_LOG" \
    || echo "  FAIL (see $LOG)" | tee -a "$SWEEP_LOG"
done

echo "[done $(date +%Y%m%d_%H%M%S)]" | tee -a "$SWEEP_LOG"
