# Whale T2048 Breakthrough — 2026-04-21

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
- The single-GPU lead family became Hybrid B:
  - `DQ=128,64,8,5`
  - `DKDV=128,128,8,3`
  - `DKDV_MAXNREG=256`
- Best known exact-shape heavy point on that family:
  - `QMR=1664`
  - whale `0.2856340484172106`
  - FA3 `0.2983246239423752`

## Blessed single-GPU carry-forward point

- Carry-forward single-GPU family remains Hybrid B.
- Blessed single-GPU point:
  - `DQ=128,64,8,5`
  - `DKDV=128,128,8,3`
  - `DKDV_MAXNREG=256`
  - `QMR=1664`
- Reason this point was carried forward:
  - it owns the best known exact-shape heavy point
  - it survived late winner-lock passes at `0.287139` and `0.290260`
  - nearby `1952/2192/2240` remained strong challengers, but they showed more wobble than `1664`
- `DKDV_MR` resolution:
  - `256` is the winning carry-forward value
  - `192` was catastrophic
  - `160` was bad
  - `224` occasionally approached the baseline but did not beat `256` cleanly enough to promote

## Blessed 2x carry-forward point

- The final distributed carry-forward is primary, not Hybrid B.
- Blessed `2x` kernel point:
  - `DQ=128,64,8,5`
  - `DKDV=128,64,8,3`
  - `DKDV_MAXNREG=256`
  - `QMR=1984`
- Final head-to-head winner-lock replays:
  - whale `0.293081552297`, FA3 `0.310599344239`
  - whale `0.295900591940`, FA3 `0.307649663761`
- Strongest displaced challenger:
  - fallback `DQ=128,64,8,2 / DKDV=128,128,8,3 / DKDV_MAXNREG=256 / QMR=928`
  - final head-to-head replays:
    - whale `0.295171663806`, FA3 `0.300850479931`
    - whale `0.296852319911`, FA3 `0.312706207991`

## Current interpretation

- The committed point proved the original whale `<` FA3 crossover was real.
- Hybrid B won the single-GPU frontier cleanly enough to become the first downstream anchor.
- `2x` replay pressure then reordered the winner and selected primary `1984` as the distributed carry-forward.
- The current downstream job is no longer `2x` validation.
- The next job is whale-donor integration using the blessed `2x` primary anchor.
