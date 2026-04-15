# RESULTS

**RUNNING**

## Summary
- Hypothesis: clean offline post-build compression/eval on the locked Midnight_Anchor checkpoint.
- Result: run launched from the saved anchor checkpoint and is currently in flight on the pod.
- Delta vs build artifact: pending completion of quant roundtrip and deployed sliding eval.

## Evidence
- source checkpoint: `/workspace/SOTA_FINAL/artifacts/midnight_anchor/final_model_anchor.pt`
- seed 444 log: `legs/2026-04-15_midnight_anchor_compress/logs/full_seed444_20260415_171733.log`
- artifact size: pending

## Verdict
- RUNNING
- What to carry forward: locked float anchor as the source of truth.
- What to avoid next: mixing build changes with compression changes in the same lane.
