# Whale T2048 Next Gates — 2026-04-19

This note keeps the downstream path fresh while single-GPU kernel work remains primary.

## Current single-GPU frontier

- Lead family: `DQ=128,64,8,5 / DKDV=128,128,8,3 / DKDV_MAXNREG=256`
- Best known exact-shape point so far:
  - `QMR=1664`
  - whale `0.2856340484172106`
  - FA3 `0.2983246239423752`

## What still needs to be closed on single GPU

- Lower edge below `1664`
- Upper edge above `2304`
- `DKDV_MR` interaction on the winning family
- Direct challenger pressure from:
  - primary `DQ=128,64,8,5 / DKDV=128,64,8,3`
  - fallback `DQ=128,64,8,2 / DKDV=128,128,8,3`
  - hybrid A `DQ=128,64,8,2 / DKDV=128,64,8,3`

## Downstream path to keep ready

Readiness mechanism:
- Keep one machine-readable winner anchor for the current carry-forward family.
- Downstream scripts should read that anchor instead of hardcoding the current winner in multiple places.
- The anchor can remain provisional until single-GPU bracketing is complete.

### 1. Multi-GPU signal

Purpose:
- verify whether the single-GPU ordering survives distributed / end-to-end pressure

What should be ready:
- the current winning family
- one strong challenger family
- a clean per-rank baseline against FA3

Do not promote to this step until:
- the single-GPU winning family is bracketed tightly enough that the carry-forward choice is not arbitrary

### 2. Whale-donor integration

Purpose:
- carry the strongest single-GPU kernel family into the donor path once the family choice is stable

What should be ready:
- winner family spec
- challenger spec
- exact-shape evidence references
- a short rationale for why the chosen family outranks the others on absolute whale time

## Current readiness stance

- Keep both downstream gates ready to execute within hours once single-GPU bracketing is tight enough.
- Do not start either gate early just because the current family is green.
- The single-GPU frontier is already winning; the remaining work is about choosing the right winner to carry forward, not about proving whale can ever beat FA3.
