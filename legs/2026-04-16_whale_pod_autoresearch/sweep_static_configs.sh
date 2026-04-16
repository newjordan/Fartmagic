#!/usr/bin/env bash
# Static config sweep: force a single (BLOCK_M, BLOCK_N, warps, stages) for each
# of the three kernels via env vars, run the bench, parse headline mean_ms.
#
# Outputs one row per config to EVID/static_sweep.tsv.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LEG=legs/2026-04-16_whale_pod_autoresearch
EVID="$LEG/evidence"
TSV="$EVID/static_sweep.tsv"
mkdir -p "$EVID"

echo -e "kernel\tBM\tBN\twarps\tstages\tfwd_ms\tfwd_bwd_ms\tfwd_err" > "$TSV"

# Headline shape for the sweep.
SHAPE="${SHAPE:-4,2048,8,4,64}"

# Candidate configs to sweep. Keep it tight - full cross product would be 72 runs.
CONFIGS=(
  "64,64,4,3"    # autotune-picked baseline
  "64,64,4,2"
  "64,64,8,3"
  "128,64,4,3"
  "128,64,8,3"
  "128,64,8,4"
  "128,128,4,3"
  "128,128,8,3"
  "128,128,8,4"
  "64,128,4,3"
  "64,128,8,3"
  "64,128,8,4"
)

# First: vary FWD only, keep bwd at 64,64,4,3
echo "=== FWD sweep ==="
for CFG in "${CONFIGS[@]}"; do
  IFS=',' read -r BM BN W S <<<"$CFG"
  OUT="$EVID/static_fwd_${BM}_${BN}_${W}_${S}.json"
  PYTHONPATH=. \
    WHALE_FWD_CONFIG="$CFG" \
    WHALE_BWD_KV_CONFIG="64,64,4,3" \
    WHALE_BWD_Q_CONFIG="64,64,4,3" \
    /venv/main/bin/python3 -u "$LEG/bench_head_to_head.py" \
      --out "$OUT" \
      --label "static_fwd_${CFG}" \
      --shapes "$SHAPE" \
      --backends whale \
      --warmup 10 --iters 40 2>&1 | grep "^whale" || true
  FWD=$(/venv/main/bin/python3 -c "import json; r=json.load(open('$OUT'))['results'][0]; print(f\"{r['fwd']['mean_ms']:.4f}\\t{r['fwd_bwd']['mean_ms']:.4f}\\t{r['err']['fwd_max_abs']}\")")
  echo -e "fwd\t$BM\t$BN\t$W\t$S\t$FWD" >> "$TSV"
done

# Then: vary BWD_KV only (most impactful bwd kernel)
echo "=== BWD_KV sweep ==="
for CFG in "${CONFIGS[@]}"; do
  IFS=',' read -r BM BN W S <<<"$CFG"
  OUT="$EVID/static_bkv_${BM}_${BN}_${W}_${S}.json"
  PYTHONPATH=. \
    WHALE_FWD_CONFIG="64,64,4,3" \
    WHALE_BWD_KV_CONFIG="$CFG" \
    WHALE_BWD_Q_CONFIG="64,64,4,3" \
    /venv/main/bin/python3 -u "$LEG/bench_head_to_head.py" \
      --out "$OUT" \
      --label "static_bkv_${CFG}" \
      --shapes "$SHAPE" \
      --backends whale \
      --warmup 10 --iters 40 2>&1 | grep "^whale" || true
  FWD=$(/venv/main/bin/python3 -c "import json; r=json.load(open('$OUT'))['results'][0]; print(f\"{r['fwd']['mean_ms']:.4f}\\t{r['fwd_bwd']['mean_ms']:.4f}\\t{r['err']['fwd_max_abs']}\")")
  echo -e "bkv\t$BM\t$BN\t$W\t$S\t$FWD" >> "$TSV"
done

# Then: vary BWD_Q only
echo "=== BWD_Q sweep ==="
for CFG in "${CONFIGS[@]}"; do
  IFS=',' read -r BM BN W S <<<"$CFG"
  OUT="$EVID/static_bq_${BM}_${BN}_${W}_${S}.json"
  PYTHONPATH=. \
    WHALE_FWD_CONFIG="64,64,4,3" \
    WHALE_BWD_KV_CONFIG="64,64,4,3" \
    WHALE_BWD_Q_CONFIG="$CFG" \
    /venv/main/bin/python3 -u "$LEG/bench_head_to_head.py" \
      --out "$OUT" \
      --label "static_bq_${CFG}" \
      --shapes "$SHAPE" \
      --backends whale \
      --warmup 10 --iters 40 2>&1 | grep "^whale" || true
  FWD=$(/venv/main/bin/python3 -c "import json; r=json.load(open('$OUT'))['results'][0]; print(f\"{r['fwd']['mean_ms']:.4f}\\t{r['fwd_bwd']['mean_ms']:.4f}\\t{r['err']['fwd_max_abs']}\")")
  echo -e "bq\t$BM\t$BN\t$W\t$S\t$FWD" >> "$TSV"
done

echo "=== TOP 5 per kernel (by fwd_bwd_ms) ==="
for K in fwd bkv bq; do
  echo "--- $K ---"
  grep "^$K" "$TSV" | sort -k7 -g | head -5
done
echo ""
echo "Wrote $TSV"
