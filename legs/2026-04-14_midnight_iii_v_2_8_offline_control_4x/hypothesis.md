# Hypothesis — 2026-04-14_midnight_iii_v_2_8_offline_control_4x

Parent: legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu/train_gpt.py

## One Variable
- Name: `GPTQ_CAL_MODE`
- New value: `offline`
- Baseline value: `online_cpu`

## Why
- Once recursion activates, this control tells us whether online Hessian collection is still harming the warmdown path even when the rest of the III.V recipe is unchanged.
- This gives tonight's sweep a clean algorithmic baseline against the online runs on the same 4xH100 600s budget.

## Gate Pass Criteria
- Full-run evidence is authoritative for this leg; gate is sanity-only.
- 4xH100 1200s run must provide a direct post-EMA and final exact comparison against the online legs.
