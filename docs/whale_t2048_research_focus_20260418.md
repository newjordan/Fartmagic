# Whale T2048 Research Focus — 2026-04-19

This document captures the current research doctrine for the whale kernel at the competition shape `(2,2048,8,4,64)`.

## Confidence rule

- Primary confidence vector: absolute whale `fb_mean_ms`
- Secondary confidence vector: absolute FA3 `fb_mean_ms`
- Supporting signal only: whale-vs-FA3 delta

## Current frontier

- **Frontier / Hybrid B:** `DQ=128,64,8,5` + `DKDV=128,128,8,3` + `DKDV_MAXNREG=256`
  - Current best known point: `QMR=1664`, whale `0.285634048417`, FA3 `0.298324623942`
  - Current measured basin: `304` heavy cases, `298/304` green, with a broad green floor spanning roughly `1664..2304`
- **Primary challenger:** `DQ=128,64,8,5` + `DKDV=128,64,8,3` + `DKDV_MAXNREG=256`
  - Still strong and still useful as a challenger family, but no longer the lead frontier
- **Fallback challenger:** `DQ=128,64,8,2` + `DKDV=128,128,8,3` + `DKDV_MAXNREG=256`
  - Still a real family and still a useful truth source for playoff checks
- **Hybrid A challenger:** `DQ=128,64,8,2` + `DKDV=128,64,8,3` + `DKDV_MAXNREG=256`
  - Still worth direct comparison against fallback and Hybrid B
- **Secondary quarantine:** `DQ=128,64,8,5` + `DKDV=256,128,8,2` + `DKDV_MAXNREG=256`
  - Keep only sparse sentinels unless it materially improves absolute whale time

## Current search policy

- GPU 0 exploits the best known Hybrid B basin floor.
- GPU 1 explores adjacent horizons and orthogonal interaction planes on the same family.
- Controlled exploration is now part of correctness, not optional garnish:
  - lower horizon below `1664`
  - upper horizon above `2304`
  - `DKDV_MR` interaction checks at strong Hybrid B points

## Stopping rule

- Kernel work continues until the peaks are mapped on all active calibrations.
- Apparent winners must be tested around their local neighborhood to rule out false peaks.
- Do not push past a bad peak; use local perturbation and replicate-heavy checks to confirm the crest.
- Do not call the surface mapped while a frontier family still has an open lower edge, upper edge, or unresolved orthogonal interaction plane.

## Downstream readiness

- Keep the downstream path fresh and ready for:
  1. multi-GPU signal measurement
  2. whale-donor integration
- Those steps remain downstream until the exact-shape single-GPU frontier is bracketed tightly enough to choose the carry-forward kernel family with confidence.
