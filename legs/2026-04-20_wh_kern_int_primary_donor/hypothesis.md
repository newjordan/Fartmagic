# Hypothesis — 2026-04-20_wh_kern_int_primary_donor

Parent donor: `2026-04-15_whale_l12_dim576_matrixbits5`

## One Variable
- Name: `best donor + blessed 2x whale kernel`
- Change: preserve the best donor recipe from `whale_l12_dim576_matrixbits5`, but replace the attention path with the blessed native whale kernel carry-forward selected by the completed 2x validation.

## Blessed kernel anchor
- Family: `primary`
- Config: `DQ=128,64,8,5 / DKDV=128,64,8,3 / DKDV_MR=256 / QMR=1984`
- Source: `legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/winner_anchor.json`

## Why
- Kernel search is closed enough to promote the 2x carry-forward winner.
- The donor branch exists to turn the best donor and the blessed whale kernel into one real runnable model path.
- Challenger kernels stay as rollback references only.

## Pass Criteria
- The runner uses `pr1493_decompressed.py` contract rather than the purged `gemini_runs/pr1493_train_gpt.py` path.
- The run logs the blessed primary kernel settings.
- The donor model compiles and runs with the native whale kernel path.
