# Hypothesis: NUM_LAYERS=12 with Brotli + Mixed-Int

## Variable
NUM_LAYERS=12 (from 11). One variable vs brotli+mixedint baseline.
MODEL_DIM=512 stays (head_dim=64, flash_attn_3 compatible).

## Why
- MODEL_DIM=528 crashed: flash_attn_3 requires head_dim multiple of 8 (528/8=66, fail)
- MODEL_DIM=576 too big: 33.8M params → ~18MB artifact (over 16MB)
- NUM_LAYERS=12 adds +7.7% params (29.1M), artifact ~15.58MB — fits with margin
- Brotli+mixed-int freed 2.23MB headroom. Spending it on depth.
- No GPTQ (SKIP_GPTQ=1) — keep full 600s training budget.

## Expected
- Params: ~29.1M (+2.1M vs baseline 27.0M)
- Artifact: ~15.58MB (fits under 16MB)
- BPB: deeper model should improve representation quality
- Steps: may be slightly fewer (~5% slower per step from extra layer)

## Gate target
Artifact fits under 16MB. BPB trending below 11-layer control.

## Parent
`experiments/2026-04-06_neural_sota_brotli_mixedint/` (1.11025 BPB, 14.44MB)
