# Hypothesis — 2026-04-14_midnight_iii_v_bank_gptq_fix

Parent: legs/2026-04-13_midnight_iii_v_qattn6/train_gpt.py

## One Variable
- Name: `bank-aware online GPTQ + deployed eval`
- Change: collect bank Hessians online during late warmdown (looping-on, low lr_mul, high progress), finalize once at training end (no TokenStream calibration stop), then GPTQ-quantize shared banks and report final sliding eval on the reloaded quantized artifact.

## Why
- The parent lane trains to good float quality but the export path reports `gptq:calibrated 0 layers`, falls back to naive bank quantization, and collapses `final_quant_roundtrip_exact`.
- The parent lane also logs `final_sliding_window_exact` on the live model, which hides deployed-artifact quality.
- The parent lane pays a hard stop for post-train GPTQ calibration; this leg moves stat collection into late warmdown.

## Pass Criteria
- Logs show `gptq_online:start` during late warmdown and `gptq:online finalized ... samples:...` at reconcile.
- No end-of-run `gptq:calibrating with training data...` TokenStream pass.
- `final_quant_roundtrip_exact` materially improves versus the parent leg.
- `final_sliding_window_exact` is measured on the reloaded quantized artifact and remains submission-legal on bytes.
