# Whale T2048 Research Focus — 2026-04-20

This document captures the current research doctrine for the whale kernel at the competition shape `(2,2048,8,4,64)`.

## Confidence rule

- Primary confidence vector: absolute whale `fb_mean_ms`
- Secondary confidence vector: absolute FA3 `fb_mean_ms`
- Supporting signal only: whale-vs-FA3 delta

## Current frontier

- **Blessed single-GPU carry-forward / Hybrid B:** `DQ=128,64,8,5` + `DKDV=128,128,8,3` + `DKDV_MAXNREG=256`
  - Carry-forward `QMR=1664`
  - Best known exact-shape heavy point: whale `0.285634048417`, FA3 `0.298324623942`
  - Winner-lock rechecks kept `1664` low and green at `0.287139` and `0.290260`
  - Interpretation: `1664` is the cleanest practical anchor even though nearby `1952/2192/2240` also showed strong upside
- **Nearest same-family challengers:** `QMR=1952`, `QMR=2192`, `QMR=2240`
  - These remain the highest-value same-family challengers for `2x` confirmation because they posted low absolute whale times but more wobble than `1664`
- **Primary challenger family:** `DQ=128,64,8,5` + `DKDV=128,64,8,3` + `DKDV_MAXNREG=256`
  - Still a valid challenger family, but no longer the promoted carry-forward line
- **Fallback challenger family:** `DQ=128,64,8,2` + `DKDV=128,128,8,3` + `DKDV_MAXNREG=256`
  - Still a useful truth source and backup family, but it is no longer the promoted line
- **Hybrid A challenger family:** `DQ=128,64,8,2` + `DKDV=128,64,8,3` + `DKDV_MAXNREG=256`
  - Still useful for compact downstream comparison only
- **Secondary quarantine:** `DQ=128,64,8,5` + `DKDV=256,128,8,2` + `DKDV_MAXNREG=256`
  - Keep out of the frontier unless a downstream contradiction reopens it

## Current search policy

- Single-GPU frontier expansion is no longer the lead job.
- The current outcome of single-GPU work is:
  - broad Hybrid B basin found
  - no deeper minimum found outside the mid-band
  - `DKDV_MR=256` confirmed as the correct carry-forward setting
- Downstream work should now treat Hybrid B `QMR=1664` as the promoted anchor and use `1952/2192/2240` as compact challengers.

## Stopping rule

- For the exact-shape single-GPU kernel bench, the surface is now mapped tightly enough to promote a winner.
- Reopen single-GPU kernel search only if:
  - `2x` signal disagrees with the promoted anchor
  - the donor integration exposes a contradiction
  - a challenger family materially outruns the blessed anchor under downstream pressure

## Downstream readiness

- The next gates are now active, not hypothetical:
  1. `2x` / multi-GPU signal measurement
  2. whale-donor integration
- Carry-forward anchor:
  - family: Hybrid B
  - `DQ=128,64,8,5`
  - `DKDV=128,128,8,3`
  - `DKDV_MAXNREG=256`
  - `QMR=1664`
