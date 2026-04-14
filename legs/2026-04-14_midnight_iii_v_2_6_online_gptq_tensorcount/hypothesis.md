# Hypothesis — 2026-04-14_midnight_iii_v_2_6_online_gptq_tensorcount

Parent: legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu/train_gpt.py

## One Variable
- Name: `ONLINE_GPTQ_N_SEEN_STORAGE`
- New value: `scalar tensor counters with in-place tensor add`
- Baseline value: `python int counters mutated inside compiled forward`

## Why
- `v_2_3` hit `torch._dynamo` recompilation on `self.n_seen[...] += ...` once recursion activated, then burned most of the 600s budget recompiling instead of training.
- Keeping the online GPTQ counters as tensors should remove the static-int invalidation path while preserving the same online warmdown recipe.

## Gate Pass Criteria
- Full-run evidence is authoritative for this leg; gate is sanity-only.
- 4xH100 1200s run must avoid `recompile_limit`, materially exceed `v_2_3` step 433, and keep online GPTQ active through the recursion regime.
