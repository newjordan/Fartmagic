#!/usr/bin/env bash
# Sweep forced configs on the dkdv_inline_delta kernel at the headline shape.
# Also sweeps maxnreg via WHALE_BWD_KV_MAXNREG.
#
# This sweep sets BWD_KV_CONFIG to force a single config and leaves BWD_Q to
# autotune. It times fwd+bwd over the headline shape with fused_delta variant.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT_TSV="legs/2026-04-16_whale_bwd_ablations/evidence/dkdv_inline_sweep.tsv"
mkdir -p "$(dirname "$OUT_TSV")"
echo -e "BM\tBN\twarps\tstages\tmaxnreg\twhale_fb_ms\twhale_fb_std\tfa3_fb_ms" > "$OUT_TSV"

# Candidate configs (bm,bn,warps,stages)
CONFIGS=(
  "64,64,4,2"
  "64,64,4,3"
  "64,64,4,4"
  "64,64,8,3"
  "64,128,4,3"
  "64,128,4,4"
  "64,128,8,3"
  "128,64,4,3"
  "128,64,8,3"
  "128,128,4,3"
  "128,128,8,3"
)
# Maxnreg options (0 = unset)
MAXNREGS=(0 128 160 192 224)

for cfg in "${CONFIGS[@]}"; do
  IFS=',' read -r BM BN W S <<<"$cfg"
  for MR in "${MAXNREGS[@]}"; do
    TAG="bm${BM}_bn${BN}_w${W}_s${S}_mr${MR}"
    OUTFILE="legs/2026-04-16_whale_bwd_ablations/evidence/dkdv_inline_${TAG}.json"
    echo "=== $TAG ==="
    MR_ARG=""
    if [[ "$MR" != "0" ]]; then
      export WHALE_BWD_KV_MAXNREG="$MR"
    else
      unset WHALE_BWD_KV_MAXNREG || true
    fi
    WHALE_BWD_VARIANT=fused_delta \
    WHALE_BWD_KV_CONFIG="$cfg" \
      PYTHONPATH=. /venv/main/bin/python3 -u legs/2026-04-16_whale_bwd_ablations/bench_stable.py \
        --out "$OUTFILE" \
        --label "$TAG" \
        --shape "4,2048,8,4,64" \
        --backends "whale_fast,fa3" \
        --rounds 6 --iters 80 2>&1 | tail -5

    if [[ -f "$OUTFILE" ]]; then
      WH=$(/venv/main/bin/python3 -c "import json;d=json.load(open('$OUTFILE'));r=d['results'][0]['backends']['whale_fast'];print(f\"{r['fb_mean_ms']:.3f}\t{r['fb_std_ms']:.3f}\")")
      FA=$(/venv/main/bin/python3 -c "import json;d=json.load(open('$OUTFILE'));r=d['results'][0]['backends']['fa3'];print(f\"{r['fb_mean_ms']:.3f}\")")
      echo -e "${BM}\t${BN}\t${W}\t${S}\t${MR}\t${WH}\t${FA}" >> "$OUT_TSV"
    fi
  done
done
echo ""
echo "=== Results (sorted by whale_fb_ms) ==="
head -1 "$OUT_TSV"
tail -n +2 "$OUT_TSV" | sort -k6 -n | head -15
