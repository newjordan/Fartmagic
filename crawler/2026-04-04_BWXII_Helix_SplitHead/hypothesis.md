# BW XII — Helix SplitHead

## Architecture
The crawler is a cross-referencing correction engine, not a transformer.
All attention heads cross-attend to the flat stream. No self-attention.
Position-agnostic K projections (no RoPE on cross keys) for content routing.

## Changes vs Ouroboros (BW XI)
1. CRAWLER_CROSS_HEADS=4 — full cross-attention to flat stream (−0.027 micro)
2. MUON_WD=0.12 — higher weight decay for quant-friendly shared weights (−0.012 micro)
3. Combined micro: 1.8249 BPP (−0.086 vs no helix baseline)

## Config
- 9F flat + 1 crawler × 1 loop
- HELIX=1, HELIX_DIM=192, HELIX_STRIDE=1
- CRAWLER_CROSS_HEADS=4 (full cross-attention)
- MUON_WD=0.12
- SKIP_GPTQ=0, LOOP_AWARE_GPTQ=1
- QK_GAIN_INIT=4.0, brotli compression
- COMPILE_FULLGRAPH=1

## Gate target
Beat Ouroboros (1.13727008) on int6_sw_bpb at seed=444.
