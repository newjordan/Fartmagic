# Hypothesis — 2026-04-10_qk_gain_525

## One Variable
- Name: `QK_GAIN_INIT`
- New value: `5.25`
- Baseline value: `1.5`

## Parent
- Source: `vault/train_gpt_midnight_12l_sota_REAL.py`
- Copy: `legs/2026-04-10_qk_gain_525/train_gpt.py`

## Why
- Bridge evidence already showed monotonic improvement from `1.5 -> 5.0 -> 5.25`.
- This changes only attention sharpness; it does not require architecture edits, stack edits, or a new export path.
- It is the cleanest first promotion attempt because the locked source stays intact and the copied leg changes exactly one runtime variable.
- The copied test file now owns the screen profile defaults directly; the runner only launches it.

## Gate Pass Criteria
- 4xGPU, 6-minute screen improves versus control on seed `444`.
- Primary signal: lower final `val_bpb` at the 6-minute stop than the copied-source control.
- If no clear improvement: stop and mark DOES NOT PROMOTE.
