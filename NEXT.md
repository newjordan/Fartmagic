# Next

1. Freeze `legs/2026-04-13_midnight_iii_v_qattn6` as promoted reference.
2. Run post-quant sweep child legs only (one variable each, no seed churn first).
3. Keep all sweep changes in each leg's `tracked_env.sh` or `train_gpt.py`.
4. Run `python3 scripts/leg_diff_guard.py legs/<leg>` before each gate/run.
5. Record each sweep in that leg's `ablation.md` and `RESULTS.md` with exact log line citations.
