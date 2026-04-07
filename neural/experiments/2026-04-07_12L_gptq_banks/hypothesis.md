# Hypothesis: GPTQ Bank-Aware Calibration on 12L Model

## Variable
GPTQ calibration fixed to instrument F.linear calls for bank tensors.
ONE variable: GPTQ quality vs naive quantization (SKIP_GPTQ=0 vs 1).

## Problem
GPTQ calibration only hooked nn.Linear/CastedLinear modules (2 total: bigram.proj etc).
The main weights live in 3D bank tensors (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank)
used via F.linear — no modules, no hooks, no Hessians collected.
Result: "gptq:calibrated 2 layers" = GPTQ did nothing for 99%+ of parameters.

## Fix
1. `gptq_calibrate`: monkey-patch F.linear during calibration to capture inputs
   for each bank slice (keyed as `bank_name.slice_idx`). Uses data_ptr matching.
2. `mixed_quantize_gptq`: detect 3D bank tensors, apply per-slice GPTQ using
   the per-slice Hessians. Cat results to match existing dequant format.

## Expected
- ~72 bank slice Hessians collected (4 banks × ~18 slices avg)
- GPTQ should reduce the roundtrip quant gap (currently 0.014 BPB)
- Trade-off: 45s calibration reserve (GPTQ_RESERVE_MS=45000) = ~500 fewer training steps
- Net: GPTQ quality gain > step loss IF gap closes by >0.005

## Gate target
- "gptq:calibrated N layers" should show N >> 2 (expect ~72 bank + 2 module = 74)
- Roundtrip BPB should drop below 1.14 (from 1.1457 naive)
- Artifact size similar (~15.5MB)

## Parent
`experiments/2026-04-07_dim528_brotli_mixedint/` (12L, 1.10819 BPB, 15.57MB)
