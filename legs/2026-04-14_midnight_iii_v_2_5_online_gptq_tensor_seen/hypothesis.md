# Hypothesis — 2026-04-14_midnight_iii_v_2_3_online_gptq_cpu

Parent: legs/2026-04-13_midnight_iii_v_2_1_compile_lock/train_gpt.py

## One Variable
- Name: `GPTQ_CAL_MODE`
- New value: `online_cpu`
- Baseline value: `offline`

## Why
- Offline GPTQ calibration consumes dedicated end-of-run wallclock and can miss late-recursion dynamics.
- Online CPU Hessian accumulation during training should preserve effective training budget while tracking the active looping regime.

## Gate Pass Criteria
- Log shows `gptq:online` collection activity during training.
- Final section reports `gptq:calibrated` with non-zero layers.
- Report required diagnostics and exact final metrics with submission bytes.
