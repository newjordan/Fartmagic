# Whale T2048 Next Gates — 2026-04-20

This note keeps the downstream path fresh now that the single-GPU kernel winner is blessed.

## Current single-GPU winner

- Blessed carry-forward family: `DQ=128,64,8,5 / DKDV=128,128,8,3 / DKDV_MAXNREG=256`
- Blessed carry-forward point:
  - `QMR=1664`
  - best known exact-shape heavy point: whale `0.2856340484172106`, FA3 `0.2983246239423752`
  - winner-lock rechecks kept `1664` low and green at `0.287139` and `0.290260`
- Same-family `2x` challengers to keep ready:
  - `QMR=1952`
  - `QMR=2192`
  - `QMR=2240`

## Single-GPU closeout status

- Broad basin established.
- No deeper minimum was found outside the mid-band.
- `DKDV_MR=256` won the interaction plane cleanly enough to carry forward.
- Additional single-GPU work should now be treated as reopen-only, not default work.

## Downstream path to keep ready

Readiness mechanism:
- Keep one machine-readable winner anchor for the current carry-forward family.
- Downstream scripts should read that anchor instead of hardcoding the current winner in multiple places.
- The anchor is now promoted beyond provisional status for downstream use.

### 1. Multi-GPU signal

Purpose:
- verify whether the blessed single-GPU ordering survives `2x` / distributed pressure

What should be ready:
- anchor: Hybrid B `QMR=1664`
- same-family challengers: `1952`, `2192`, `2240`
- one clean per-rank FA3 baseline
- optional family challengers:
  - primary `DQ=128,64,8,5 / DKDV=128,64,8,3`
  - fallback `DQ=128,64,8,2 / DKDV=128,128,8,3`

### 2. Whale-donor integration

Purpose:
- carry the blessed Hybrid B anchor into the donor path once `2x` signal does not contradict it

What should be ready:
- winner family spec
- same-family challenger specs
- exact-shape evidence references
- short rationale for why `1664` beat the alternatives as the carry-forward anchor

## Current readiness stance

- Single-GPU choice is now tight enough to move downstream.
- Multi-GPU signal is the next immediate gate.
- Donor integration should use the same anchor file and the same blessed family spec, rather than reopening single-GPU tuning by default.
