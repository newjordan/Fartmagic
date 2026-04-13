# Hypothesis — 2026-04-13_midnight_iii_v_2_postema_gptq

Parent: legs/2026-04-13_midnight_iii_v_1/train_gpt.py

## One Variable
- Name: `GPTQ_CALIBRATION_ORDER`
- New value: `Collect GPTQ Hessians after EMA/LAWA weights are applied, so calibration matches the exact exported model weights`
- Baseline value: `Collect GPTQ Hessians before EMA/LAWA averaging, then quantize averaged weights with pre-EMA Hessians`

## Why
- Current logs show large divergence between `DIAGNOSTIC post_ema` and `final_quant_roundtrip_exact`.
- Post-EMA calibration should reduce Hessian/weight mismatch during GPTQ quantization without changing architecture or quant policy bits.

## Gate Pass Criteria
- 1xGPU 2000-step gate improves versus control proxy.
- If no clear improvement: stop and mark DOES NOT PROMOTE.
