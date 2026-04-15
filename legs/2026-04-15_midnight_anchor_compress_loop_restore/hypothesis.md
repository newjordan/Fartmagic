# Hypothesis — 2026-04-15_midnight_anchor_compress_loop_restore

Parent: legs/2026-04-15_midnight_anchor_compress/train_gpt.py

## One Variable
- Name: `checkpoint loop-state restore`
- Change: after loading the finished Midnight Anchor checkpoint via `INIT_MODEL_PATH`, restore `base_model.looping_active` before eval and GPTQ calibration.

## Why
- The first export-only compression leg loaded the right checkpoint but evaluated it with looping disabled.
- This keeps the same quant policy and offline flow, but restores the execution state the anchor build actually used.

## Gate Pass Criteria
- Emit `init_model:loop_state_restored`.
- Recover float parity near the anchor build `DIAGNOSTIC post_ema`.
- Emit `final_quant_roundtrip_exact` and `final_sliding_window_exact` from the reloaded quantized artifact.
