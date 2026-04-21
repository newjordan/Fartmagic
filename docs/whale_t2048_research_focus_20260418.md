# Whale T2048 Research Focus — 2026-04-21

This document captures the final kernel doctrine for the competition shape `(2,2048,8,4,64)`.

## Confidence rule

- Primary confidence vector: absolute whale `fb_mean_ms`
- Secondary confidence vector: absolute FA3 `fb_mean_ms`
- Supporting signal only: whale-vs-FA3 delta

## Final kernel outcome

- **Blessed single-GPU carry-forward / Hybrid B:** `DQ=128,64,8,5` + `DKDV=128,128,8,3` + `DKDV_MAXNREG=256`
  - Carry-forward `QMR=1664`
  - Best known exact-shape heavy point: whale `0.285634048417`, FA3 `0.298324623942`
  - Winner-lock rechecks kept `1664` low and green at `0.287139` and `0.290260`
  - Interpretation: `1664` is the cleanest practical single-GPU anchor even though nearby `1952/2192/2240` also showed strong upside
- **Blessed 2x carry-forward / Primary:** `DQ=128,64,8,5` + `DKDV=128,64,8,3` + `DKDV_MAXNREG=256`
  - Carry-forward `QMR=1984`
  - Final `2x` winner-lock replays:
    - whale `0.293081552297`, FA3 `0.310599344239`
    - whale `0.295900591940`, FA3 `0.307649663761`
  - Interpretation: primary won the final apples-to-apples `2x` head-to-head over fallback and became the distributed carry-forward
- **Strongest displaced 2x challenger / Fallback:** `DQ=128,64,8,2` + `DKDV=128,128,8,3` + `DKDV_MAXNREG=256`
  - Best earlier `2x` flash: `QMR=928`, whale `0.285529936209`, FA3 `0.306176656052`
  - Final `2x` winner-lock replays came in slower than primary:
    - whale `0.295171663806`, FA3 `0.300850479931`
    - whale `0.296852319911`, FA3 `0.312706207991`
- **Hybrid A challenger family:** `DQ=128,64,8,2` + `DKDV=128,64,8,3` + `DKDV_MAXNREG=256`
  - Useful as the last insurance lane in `2x`, but slower than the final primary carry-forward
- **Secondary quarantine:** `DQ=128,64,8,5` + `DKDV=256,128,8,2` + `DKDV_MAXNREG=256`
  - Keep out of the frontier unless a downstream contradiction reopens it

## Search status

- Single-GPU frontier expansion is done.
- `2x` playoff is done.
- The current outcome of kernel work is:
  - broad single-GPU Hybrid B basin found and mapped tightly enough to bless `1664`
  - `2x` distributed signal reordered the winner and selected primary `1984`
  - `DKDV_MR=256` remained the correct carry-forward setting through the closeout work
- Downstream work should now treat primary `QMR=1984` as the promoted anchor.

## Stopping rule

- Kernel search is closed unless downstream contradiction forces a reopen.
- Reopen kernel work only if:
  - donor integration exposes a contradiction
  - end-to-end training pressure materially overturns the `2x` ordering
  - a preserved challenger materially outruns primary `1984` under a new gate

## Downstream readiness

- `2x` / multi-GPU signal measurement is complete.
- The next gate is whale-donor integration.
- Carry-forward anchor:
  - family: primary
  - `DQ=128,64,8,5`
  - `DKDV=128,64,8,3`
  - `DKDV_MAXNREG=256`
  - `QMR=1984`
