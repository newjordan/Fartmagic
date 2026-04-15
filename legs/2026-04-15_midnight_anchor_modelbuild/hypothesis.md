# Hypothesis — 2026-04-15_midnight_anchor_modelbuild

Parent: legs/2026-04-13_midnight_iii_v_qattn6/train_gpt.py

## Change
- Freeze the strongest valid Midnight build recipe into a model-build-only anchor.
- Stop immediately after `DIAGNOSTIC post_ema`.

## Pass Criteria
- Training matches the qattn6 build recipe.
- The log contains `DIAGNOSTIC post_ema`.
- The log ends with `anchor:modelbuild_complete` and never reaches quant/export/final sliding.
