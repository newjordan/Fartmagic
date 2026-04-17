#!/usr/bin/env bash
# Sweep dq config space via WHALE_BWD_Q_CONFIG env override.
# Must be run AFTER dkdv sweep — reads dkdv winner from env var DKDV_WINNER.
set -euo pipefail
cd /workspace/SOTA_FINAL/legs/2026-04-17_whale_dq_wide_autotune_t2048
source ./tracked_env.sh

DKDV_WINNER="${DKDV_WINNER:-128,64,8,2}"
export WHALE_BWD_KV_TMA_CONFIG="${DKDV_WINNER}"
export WHALE_BWD_KV_TMA_MAXNREG=224

TS="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="logs/sweep_${TS}.log"
mkdir -p logs evidence
BENCH="/workspace/SOTA_FINAL/legs/2026-04-16_whale_bwd_ablations/bench_stable.py"

# (BM,BN,warps,stages,maxnreg)
CONFIGS=(
  "128,64,4,3,224"   "128,64,8,3,224"
  "128,64,8,2,224"   "128,64,8,4,224"
  "128,64,8,3,192"   "128,64,8,3,160"
  "128,128,8,3,224"  "128,128,8,3,192"
  "256,64,4,3,224"   "256,64,8,3,224"
  "64,64,4,3,224"    "64,128,8,3,224"
  "128,256,8,3,192"  "128,64,8,5,224"
)

echo "[$TS] dq wide sweep — ${#CONFIGS[@]} configs, dkdv locked at ${DKDV_WINNER}" | tee -a "$SWEEP_LOG"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tee -a "$SWEEP_LOG"

for cfg in "${CONFIGS[@]}"; do
  # Split into tile and maxnreg parts
  IFS=',' read -ra PARTS <<< "$cfg"
  TILE="${PARTS[0]},${PARTS[1]},${PARTS[2]},${PARTS[3]}"
  MAXNREG="${PARTS[4]}"
  SAFE="${cfg//,/_}"
  RUN_TS="$(date +%Y%m%d_%H%M%S)"
  OUT="evidence/dq_${SAFE}_${RUN_TS}.json"
  LOG="logs/dq_${SAFE}_${RUN_TS}.log"
  echo "[$(date +%H:%M:%S)] --> dq=${TILE} mr=${MAXNREG}  out=${OUT}" | tee -a "$SWEEP_LOG"
  rm -rf /root/.triton/cache 2>/dev/null || true
  WHALE_BWD_Q_CONFIG="${TILE}" \
  WHALE_BWD_Q_MAXNREG="${MAXNREG}" \
    /venv/main/bin/python3 "$BENCH" \
      --label "dq_${SAFE}" \
      --shape "2,2048,8,4,64" \
      --backends whale_fast,fa3 \
      --rounds 5 --iters 100 \
      --out "$OUT" >>"$LOG" 2>&1 \
    && echo "  ok" | tee -a "$SWEEP_LOG" \
    || echo "  FAIL (see $LOG)" | tee -a "$SWEEP_LOG"
done

echo "[done $(date +%Y%m%d_%H%M%S)]" | tee -a "$SWEEP_LOG"
