# Hypothesis — 2026-04-13_midnight_iii_v_2_1_compile_lock

Parent: legs/2026-04-13_midnight_iii_v_2_postema_gptq/train_gpt.py

## One Variable
- Name: `COMPILE_FULLGRAPH`
- New value: `0` (in `tracked_env.sh`)
- Baseline value: `unset/default compiler behavior from parent leg`

## Why
- Recent failures were `torch._dynamo.exc.FailOnRecompileLimitHit` under fullgraph compile during post-EMA diagnostics.
- Locking `COMPILE_FULLGRAPH=0` is a surgical runtime toggle intended to preserve parent model logic while avoiding that compile failure mode.

## Gate Pass Criteria
- Gate/full run completes past post-EMA diagnostic section without compile-limit abort.
- Preserve parent post-EMA GPTQ calibration order and report exact diagnostic and final exactness lines.
