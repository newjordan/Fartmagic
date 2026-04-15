# RESULTS

**DOES NOT PROMOTE**

## Summary
- Hypothesis: 3 physical layers at dim=1024 looped 4× = 12 effective layers. Same params as 12L, 2× hardware utilization.
- Result: Pre-quant sliding window (1.1448 BPB) narrowly met ≤1.15 target but quant roundtrip was catastrophic (2.2994 BPB). Step avg 225ms — 2× slower than 12L (105ms).
- Delta vs 12L clean (1.10898852): +0.0358 BPB pre-quant, +0.706 BPB post-quant

## Evidence
- seed 444: final_sliding_window_exact=1.14479486 final_quant_roundtrip_exact=2.29939048 post_ema=1.1685 step_avg=224.99ms steps=2535/20000 (wallclock cap 570s)
- seed 300: not run — seed 444 failed pass criteria
- artifact size: 9,117,626 bytes (mixed+brotli)

## Verdict
- DOES NOT PROMOTE
- What to carry forward: wider dims do improve pre-quant representation quality (sliding window 1.145 vs 1.109 — only 0.036 gap despite 2× fewer steps). The learning-per-FLOP is arguably better.
- What to avoid next: looped architectures with few physical layers are catastrophic for quantization. Each quant error amplifies N× through the loop. Do not loop below 6 physical layers if quantization is required.
