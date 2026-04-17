# whale_long_t_profile — hypothesis (H6)

## Side-finding from H5

H5 sweep showed whale_fast (forced `fused_delta`) loses 2× to FA3 at
T≥4096:

| shape (B,T,H,KV,D) | fused_delta fb ms | FA3 fb ms | gap   |
|:-------------------|------------------:|----------:|------:|
| 4,4096,8,4,64      | 1.092             | 0.594     | +84%  |
| 2,8192,8,4,64      | 2.065             | 1.030     | +100% |

This is a much bigger gap than the H2 (TMA dkdv) headline-shape work
would close. Pivot to it first.

The ablations leg already hinted at this: `auto` falls back from
`fused_delta` to `baseline` at T>3072, and even `auto` was 52–63% slower
than FA3 at T=4096/8192 (see `legs/2026-04-16_whale_bwd_ablations/RESULTS.md`).

## Goal

Two-phase investigation:

**Phase 1 — variant confirmation.** Run `auto`, `baseline`, `fused_delta`
on T∈{4096, 8192} headline-style shape. Confirm `auto` (which means
`baseline` at long T) is the best whale variant at long T. This rules
out "we just picked the wrong variant in H5."

**Phase 2 — kernel breakdown.** kineto-profile the best whale variant
against FA3 at T=8192. Identify which kernel(s) account for the gap:
fwd, dkdv, dq, or preprocess. The fix differs by kernel:
- fwd dominant → larger BLOCK_M / num_warps tuning, or `WHALE_FWD_VARIANT=tma`
- dkdv/dq dominant → wider KV blocks; possibly TMA on long-T path
- preprocess dominant → keep inline-Δ even at long T

## Success criteria

Phase 1: write down the best whale variant at T=8192 (with std).
Phase 2: produce a per-kernel ms breakdown for whale-best vs FA3 at
(2, 8192, 8, 4, 64).

Stretch (not in this leg): close >50% of the long-T gap.

## Variants and constraints

- `WHALE_BWD_VARIANT` ∈ {`auto`, `baseline`, `fused_delta`}.
- Single-GPU only — multi-GPU is the next phase after long-T closes.
- Same iters/rounds discipline as H5 (rounds=15, iters=300) for the
  bench. Profile uses fewer iters (~50) since kineto trace is the goal,
  not statistics.

## What this does NOT test

- TMA on dkdv (H2 — still staged, will run after long-T closes if it
  still looks worthwhile).
- Persistent fused dkdv+dq (H5 — ruled out).
- New autotune configs targeted at long T (will be a follow-up leg if
  Phase 2 fingers dkdv/dq).
