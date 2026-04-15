# RESULTS

**COMPLETE**

## Summary
- Hypothesis: clean offline post-build compression/eval on the locked Midnight_Anchor checkpoint.
- Result: failed as a backend baseline because checkpoint load left looping disabled.
- Delta vs build artifact: float parity collapsed from `1.1055` to `1.4797` before quantization.

## Evidence
- source checkpoint: `/workspace/SOTA_FINAL/artifacts/midnight_anchor/final_model_anchor.pt`
- seed 444 log: `legs/2026-04-15_midnight_anchor_compress/logs/full_seed444_20260415_171733.log`
- `DIAGNOSTIC post_ema val_bpb: 1.4797`
- `final_quant_roundtrip_exact val_bpb: 1.50248663`
- `final_sliding_window_exact val_bpb: 1.49009815`
- `Total submission size mixed+brotli: 15449790`

## Verdict
- FAIL
- What to carry forward: the bug report, not the metric.
- What to avoid next: any checkpoint eval/compression lane that does not explicitly restore loop state.
