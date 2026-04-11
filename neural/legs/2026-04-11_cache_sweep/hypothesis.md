# Hypothesis — cache_sweep

## One Variable
- Name: `COPRIME_MAX_LOADED_SHARDS`
- Arms: 4 (control), 8, 12, 24
- Baseline value: `4` (champ log `records/track_10min_16mb/2026-04-07_Midnight_12L_8xH100/train_seed444.log:26` shows `cache:4`)

## Parent
- Source: `vault/train_gpt_midnight_iii_base.py`
- SHA256: `4d265579556279e3b0d652abf078fe762117227cd2408c9eca1afd81bdb15365`
- train_gpt.py is IDENTICAL to vault (zero diff, diff guard PASS). Change is env-only in run scripts.

## Why
Cache controls how many data shards the coprime loader keeps in CPU RAM. More cached shards
= fewer disk reads during training. Each shard costs ~382 MB CPU RAM (int32, CPU-side).

Prior data (`junkyard/experiments/Rascal_Stripper_Skipgram_2200/notes/2026-03-31_next_single_gpu_pack_seed444.md`):
cache=4 vs cache=1 on single H100, 1200 steps: -3.41ms/step, -0.0011 BPB.

No data exists for cache > 4. Memory cost is trivial (cache=24 = ~9 GB CPU RAM on a
200+ GB system). Signal is step_avg_ms — faster steps = more steps in 600s wallclock.

## Screen Setup
- 4xH100, 600 steps, seed=444
- Arms: run_cache4_control.sh, run_cache8.sh, run_cache12.sh, run_cache24.sh
- Metrics: step_avg_ms (primary), post_ema val_bpb (secondary)
