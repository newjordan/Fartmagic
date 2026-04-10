# midnight_GPTQ — GPTQ Bank Fix

## Parent
Midnight 12L (vault/train_gpt_midnight_12l_sota_REAL.py) — 1.10568 BPB seed 444

## One variable
Fix GPTQ quantization for 3D bank parameters (qo_bank, kv_bank, mlp_up_bank).

## Why
GPTQ was broken: 0 tensors quantized in every prior Midnight run because the architecture
uses parameter banks (3D tensors), not nn.Linear. The calibration hooks only attached to
Linear/CastedLinear modules, and the export path only handled 2D tensors.

This fix:
1. Adds block-level hooks (attn/mlp) to collect Hessians for bank parameters
2. Adds per-slice GPTQ quantization for 3D bank tensors (60 tensors now get GPTQ)

## Gate target
Quant gap should shrink (was 0.0137 BPB at production scale).
Arm C screen showed -27% quant gap reduction at 180s/4xGPU.

## Risk
Low. This is a bug fix, not a hypothesis. Training math is unchanged.
Only the export/quantization path changes.
