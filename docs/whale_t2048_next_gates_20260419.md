# Whale T2048 Next Gates — 2026-04-21

This note keeps the downstream path fresh now that both the single-GPU and `2x` kernel winners are blessed.

## Kernel closeout summary

- Blessed single-GPU family: `DQ=128,64,8,5 / DKDV=128,128,8,3 / DKDV_MAXNREG=256 / QMR=1664`
  - best known exact-shape heavy point: whale `0.2856340484172106`, FA3 `0.2983246239423752`
- Blessed `2x` family: `DQ=128,64,8,5 / DKDV=128,64,8,3 / DKDV_MAXNREG=256 / QMR=1984`
  - final winner-lock replays:
    - whale `0.293081552297`, FA3 `0.310599344239`
    - whale `0.295900591940`, FA3 `0.307649663761`
- Strongest displaced `2x` challenger:
  - fallback `DQ=128,64,8,2 / DKDV=128,128,8,3 / DKDV_MAXNREG=256 / QMR=928`
  - final winner-lock replays:
    - whale `0.295171663806`, FA3 `0.300850479931`
    - whale `0.296852319911`, FA3 `0.312706207991`

## Kernel closeout status

- Broad basin established.
- No deeper minimum was found outside the mid-band on the single-GPU frontier.
- `DKDV_MR=256` won the interaction plane cleanly enough to carry forward.
- `2x` signal did not preserve the single-GPU ordering; it promoted primary `1984`.
- Additional kernel work should now be treated as reopen-only, not default work.

## Downstream path to keep ready

Readiness mechanism:
- Keep one machine-readable winner anchor for the current carry-forward family.
- Downstream scripts should read that anchor instead of hardcoding the current winner in multiple places.
- The anchor is now promoted beyond provisional and beyond single-GPU-only status for donor use.

### 1. Multi-GPU signal

Purpose:
- verify whether the blessed single-GPU ordering survives `2x` / distributed pressure

Outcome:
- complete
- winner: primary `DQ=128,64,8,5 / DKDV=128,64,8,3 / DKDV_MAXNREG=256 / QMR=1984`
- strongest displaced challenger: fallback `QMR=928`

### 2. Whale-donor integration

Purpose:
- carry the blessed `2x` primary anchor into the donor path

What should be ready:
- winner family spec
- displaced challenger specs
- exact-shape evidence references
- short rationale for why primary `1984` beat fallback `928` in the final `2x` lock block

## Current readiness stance

- Single-GPU choice is preserved as historical context, but the active carry-forward is now the `2x` winner.
- Multi-GPU signal is complete.
- Donor integration is the next immediate gate.
