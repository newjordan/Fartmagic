# Hypothesis — 2026-04-14_midnight_iii_v_2_7_online_gptq_compile_off

Parent: legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu/train_gpt.py

## One Variable
- Name: `COMPILE_ENABLED`
- New value: `0`
- Baseline value: `1`

## Why
- If recursion blowup is fundamentally a compiled-graph invalidation problem, turning compile off should preserve the online GPTQ recipe while removing the compiler as the failure amplifier.
- This is the most direct full-run control for whether remaining warmdown damage is caused by compile interaction or by the online GPTQ algorithm itself.

## Gate Pass Criteria
- Full-run evidence is authoritative for this leg; gate is sanity-only.
- 4xH100 1200s run must show whether the recursion regime remains stable when the compiler is removed from the path.
