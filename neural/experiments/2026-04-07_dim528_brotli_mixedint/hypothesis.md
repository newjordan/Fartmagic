# Hypothesis: MODEL_DIM=528 with Brotli + Mixed-Int

## Variable
MODEL_DIM=528 (from 512). Head dim: 66. One variable vs brotli+mixedint baseline.

## Why
Brotli+mixed-int (attn=5, mlp=6, embed=8) freed 2.23MB of headroom (14.44MB vs 16MB limit).
Spending that on +6% model width. Wider model = better per-step learning.
No GPTQ (SKIP_GPTQ=1) — keep full 600s training, no 30s calibration tax.

## Expected
- Artifact: ~15.3-15.5MB (fits comfortably)
- BPB: should improve vs 512 baseline by ~0.002-0.005 from extra capacity
- Steps: same (~6630) since no GPTQ overhead

## Gate target
Artifact fits under 16MB. BPB trending below 512 control at step 2000.

## Parent
`experiments/2026-04-06_neural_sota_brotli_mixedint/` (1.11025 BPB, 14.44MB)
