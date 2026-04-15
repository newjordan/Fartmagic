# Hypothesis — 12L Eval Fix

Parent: vault/train_gpt_midnight_12l_sota_REAL.py

## Change

Two bugs fixed in the eval pipeline. Zero training changes.

1. **Quant roundtrip eval now enables looping**: `eval_model.looping_active = base_model.looping_active` after loading dequantized weights. Previously the quant eval model ran with `looping_active=False`, so looped architectures evaluated fewer effective layers than they trained with.

2. **Sliding window eval now uses the quantized model**: `eval_val_sliding(args, eval_model, ...)` instead of `eval_val_sliding(args, base_model, ...)`. Previously the sliding window reported unquantized model quality, not the actual submission artifact quality.

## Why

All prior midnight runs reported incorrect final scores. The `final_quant_roundtrip_exact` was wrong for looped models, and `final_sliding_window_exact` always measured the unquantized model regardless.

## Pass Criteria

This is a measurement-only fix. The run is valid if:
1. Training converges identically to prior 12L runs (same loss curve, same step_avg).
2. `final_quant_roundtrip_exact` and `final_sliding_window_exact` now reflect the true quantized artifact quality.
3. We get a real baseline number to compare future work against.
