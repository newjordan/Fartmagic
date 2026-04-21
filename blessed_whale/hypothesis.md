# Hypothesis — Blessed Whale

Parent donor: `2026-04-15_whale_l12_dim576_matrixbits5`

## One Variable
- Name: `blessed whale donor + blessed 2x whale kernel`
- Change: preserve the strongest donor recipe, keep the preserved PR1493-family trainer contract, and replace the attention path with the blessed whale kernel carry-forward chosen by the completed 2x validation.

## Blessed anchor
- Family: `primary`
- Config: `DQ=128,64,8,5 / DKDV=128,64,8,3 / DKDV_MR=256 / QMR=1984`
- Source: `blessed_whale/blessed_whale_anchor.json`

## Pass criteria
- the runner uses `pr1493_decompressed.py`
- the run logs benchmark ID, trainer contract, attention mode, and kernel settings
- the donor compiles and runs with the native whale path in `B1`
