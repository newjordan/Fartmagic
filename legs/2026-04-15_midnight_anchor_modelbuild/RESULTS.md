# RESULTS

**COMPLETE**

## Summary
- Hypothesis: qattn6 can act as a clean Midnight_Anchor model-build donor when the leg stops after `DIAGNOSTIC post_ema`.
- Result: anchor build completed cleanly and wrote the canonical float checkpoint.
- Delta vs build artifact: this leg intentionally stops before GPTQ, compression, and final sliding eval.

## Evidence
- seed 444 log: `legs/2026-04-15_midnight_anchor_modelbuild/logs/full_seed444_20260415_170333.log`
- `step:4000/20000 val_bpb:1.1356`
- `step:4773/20000 val_bpb:1.1063`
- `DIAGNOSTIC post_ema val_bpb:1.1055`
- `anchor_model_path:./artifacts/midnight_anchor/final_model_anchor.pt`
- artifact size: `112323005` bytes model, `129555` bytes code

## Verdict
- ANCHOR_READY
