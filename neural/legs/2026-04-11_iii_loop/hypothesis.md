# Hypothesis — III_Loop

## One Variable
- Name: `SKIP_GPTQ`
- New value: `0` (enable GPTQ Hessian calibration)
- Baseline value: `1` (skip GPTQ, use naive per-row quantization)

## Parent
- Source: `vault/train_gpt_midnight_iii_base.py`
- SHA256: `4d265579556279e3b0d652abf078fe762117227cd2408c9eca1afd81bdb15365`
- train_gpt.py is IDENTICAL to vault (zero diff). Change is in gate.sh/run.sh only.

## Why
Midnight III has a catastrophic 0.45 BPB quant gap (1.106 pre-quant vs 1.554 quant roundtrip).
With SKIP_GPTQ=1, all sub-int8 weights (attn=int5, mlp=int6) get naive per-row clipping
instead of Hessian-aware error-compensated GPTQ. For a model with depth recurrence (layers
3-5 looped x2), quantization error compounds through the loop — Hessian-aware quantization
should allocate error budget to weights that matter most for the looped forward pass.

The model's looping_active flag is True at end of training (activated at step ~1977), so
gptq_calibrate() will run forward_logits() with looping active, collecting Hessians that
reflect the actual inference-time activation pattern.

Cost: ~30s of wallclock for GPTQ calibration (GPTQ_RESERVE_MS=30000), meaning ~300 fewer
training steps. This was net-negative on Midnight 12L (quant gap was small), but Midnight III's
0.45 BPB gap makes the tradeoff potentially worthwhile.

Proven on crawler track: loop-aware GPTQ fixed a 0.095 BPB quant catastrophe there.

## Gate Pass Criteria
- 2xGPU screen (360s): quant_roundtrip BPB meaningfully below 1.554 (current naive quant).
- Target: quant_roundtrip BPB < 1.15 would indicate the technique is working.
