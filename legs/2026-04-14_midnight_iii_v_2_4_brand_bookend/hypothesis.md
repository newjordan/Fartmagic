# Hypothesis — 2026-04-14_midnight_iii_v_2_4_brand_bookend

Parent: legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu/train_gpt.py

## One Variable
- Name: `RUN_BRAND_BANNER`
- New value: `1`, printing `brand_banner.braille.txt` before and after the launch
- Baseline value: banner disabled

## Why
- `train_gpt.py` and the training exports in `tracked_env.sh` match the parent leg.
- The bookend is a log marker around the process. It does not change the training math.

## Gate Pass Criteria
- With the tracked env sourced, the launcher prints `brand_banner:pre:start` before `torchrun` and `brand_banner:post:start` after it.
- A missing banner file is a fatal error when `RUN_BRAND_BANNER=1`.
