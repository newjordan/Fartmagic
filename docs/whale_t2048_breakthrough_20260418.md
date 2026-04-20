# Whale T2048 Breakthrough — 2026-04-19

Whale is under FA3 on heavy validation at the competition kernel-bench shape `(2,2048,8,4,64)`.

## Historical committed point

- `DQ_TILE=128,64,8,2`
- `Q_MAXNREG=256`
- `DKDV_TILE=128,128,8,3`
- `DKDV_MAXNREG=256`
- `evidence/manual_validate_dq_128_64_8_2_dqmr_256_dkdvmr_256_20260418_012119.json: results[0].backends.whale_fast.fb_mean_ms = 0.2985622241050005`
- `evidence/manual_validate_dq_128_64_8_2_dqmr_256_dkdvmr_256_20260418_012119.json: results[0].backends.fa3.fb_mean_ms = 0.31829385602474214`

## Current frontier beyond the original breakthrough

- The original committed point is still a valid whale `<` FA3 crossover and still matters as historical proof.
- It is no longer the lead frontier.
- The current lead family is now Hybrid B:
  - `DQ=128,64,8,5`
  - `DKDV=128,128,8,3`
  - `DKDV_MAXNREG=256`
- Current best known exact-shape heavy point on that family:
  - `QMR=1664`
  - whale `0.2856340484172106`
  - FA3 `0.2983246239423752`

## Interpretation

- The committed point proved the whale `<` FA3 crossover was real.
- The current work is now beyond “find any crossover.”
- The active job is to finish mapping the best single-GPU frontier, rule out false peaks, and only then carry the strongest family downstream.
