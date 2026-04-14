# Hypothesis — 2026-04-14_midnight_iii_v_2_2_ckpt_eval

Parent: legs/2026-04-13_midnight_iii_v_2_1_compile_lock/train_gpt.py

## One Variable
- Name: `CHECKPOINT_EVAL_MODE`
- New value: `Load built model checkpoint and run quant/eval path without additional training updates`
- Baseline value: `Short training-before-export path under wallclock cap`

## Why
- Quant-gap diagnosis for late recursion needs stable checkpoint-first evaluation, not wallclock-truncated retraining.
- This leg hardcodes checkpoint load and zero-step training in tracked env so repeated eval sweeps are deterministic and H100-only.

## Gate Pass Criteria
- Run logs include `init_model:loaded` from `/workspace/sota_nueral/final_model.pt`.
- Run emits `DIAGNOSTIC post_ema`, `final_quant_roundtrip_exact`, and `final_sliding_window_exact` with artifact bytes.
