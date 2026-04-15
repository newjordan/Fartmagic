# RESULTS

**COMPLETE**

## Summary
- Hypothesis: same offline compression path as `midnight_anchor_compress`, but with loop state restored on checkpoint load.
- Result: backend-corrected offline compression recovered float parity and cut quant roundtrip to `1.12205231`.
- Delta vs build artifact: float parity matched exactly at `1.1055`; quant roundtrip landed only `+0.01655` bpb over the anchor.

## Evidence
- source checkpoint: `/workspace/SOTA_FINAL/artifacts/midnight_anchor/final_model_anchor.pt`
- seed 444 log: `legs/2026-04-15_midnight_anchor_compress_loop_restore/logs/full_seed444_20260415_172614.log`
- `init_model:loop_state_restored looping:1`
- `DIAGNOSTIC post_ema val_bpb: 1.1055`
- `final_quant_roundtrip_exact val_bpb: 1.12205231`
- `final_sliding_window_exact val_bpb: 1.10506462`
- `Total submission size mixed+brotli: 15450196`

## Verdict
- PASS
- What to carry forward: restore loop state before any checkpoint eval/compression.
- What to avoid next: judging quant quality from a checkpoint path until float parity is proven first.
