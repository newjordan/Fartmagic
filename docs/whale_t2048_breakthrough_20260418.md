# Whale T2048 Breakthrough — 2026-04-20

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

## Blessed single-GPU carry-forward point

- Carry-forward family remains Hybrid B.
- Blessed carry-forward kernel point:
  - `DQ=128,64,8,5`
  - `DKDV=128,128,8,3`
  - `DKDV_MAXNREG=256`
  - `QMR=1664`
- Reason this point is carried forward:
  - it owns the best known exact-shape heavy point
  - it survived late winner-lock passes at `0.287139` and `0.290260`
  - nearby `1952/2192/2240` remained strong challengers, but they showed more wobble than `1664`
- `DKDV_MR` resolution:
  - `256` is the winning carry-forward value
  - `192` was catastrophic
  - `160` was bad
  - `224` occasionally approached the baseline but did not beat `256` cleanly enough to promote

## Interpretation

- The committed point proved the whale `<` FA3 crossover was real.
- The later Hybrid B work proved the crossover was not a one-off and found a better exact-shape family.
- The active job is no longer “find any crossover.”
- The current downstream job is to carry the blessed Hybrid B anchor into `2x` validation and donor integration.
