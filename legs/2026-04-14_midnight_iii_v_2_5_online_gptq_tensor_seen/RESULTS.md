# RESULTS

**PENDING**

## Summary
- Hypothesis: store online GPTQ `n_seen` counts as tensors instead of Python ints.
- Result: this checkout has no `logs/` directory for this leg.
- The previous text in this file described `legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu` and cited a log that is not in this tree.

## Evidence
- seed 444: no log in this leg.
- seed 300: no log in this leg.
- Code difference vs the parent is `OnlineGptqAccumulator.n_seen` in `train_gpt.py`.

## Verdict
- PROMOTES / DOES NOT PROMOTE: not decided. No run of this leg is recorded here.
