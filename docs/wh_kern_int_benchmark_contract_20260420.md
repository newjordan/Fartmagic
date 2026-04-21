# wh_kern_int Benchmark Contract — 2026-04-20

## Purpose

This document defines the mandatory benchmark set and the explicit fallback ladder for `wh_kern_int`.

The goal is to keep the branch on the blessed donor + kernel integration path and prevent drift back onto stale runners or ad hoc recovery paths.

## Non-Negotiable Contract

### Allowed trainer contract
- `pr1493_decompressed.py`

### Forbidden trainer paths
- `gemini_runs/pr1493_train_gpt.py`
- `scripts/train_gpt.py`

### Blessed default kernel mode
- mode: `whale_native`
- attention op: `custom_whale_attn_fwd`
- kernel family: `primary`
- `DQ=128,64,8,5`
- `DKDV=128,64,8,3`
- `DKDV_MR=256`
- `QMR=1984`

## Mandatory Benchmarks

All comparisons must use the same donor recipe and the same preserved PR1493-family trainer contract.

### B0 — Contract baseline
- donor: `whale_l12_dim576_matrixbits5`
- trainer contract: `pr1493_decompressed.py`
- attention mode: `fa3`
- purpose: establish the donor's same-contract baseline before whale replacement

### B1 — Blessed whale integration
- donor: `whale_l12_dim576_matrixbits5`
- trainer contract: `pr1493_decompressed.py`
- attention mode: `whale_native`
- kernel: blessed `primary @ 1984`
- purpose: the actual carry-forward path that must beat or justify replacing B0

### B2 — Strongest displaced kernel fallback
- donor: `whale_l12_dim576_matrixbits5`
- trainer contract: `pr1493_decompressed.py`
- attention mode: `whale_native`
- kernel challenger: `fallback @ 928`
- purpose: recovery lane if the blessed primary kernel fails under donor integration

### B3 — Hybrid diagnostic only
- donor: `whale_l12_dim576_matrixbits5`
- trainer contract: `pr1493_decompressed.py`
- attention mode: `hybrid`
- op: `whale_fwd_fa3_bwd`
- purpose: diagnose whether failure is in the native whale backward path versus the donor itself
- this is not a production carry-forward unless explicitly re-approved

## Minimum Acceptance Order

1. B0 runs successfully and exports artifacts.
2. B1 runs successfully on the same contract and donor recipe.
3. If B1 fails, run B2 before reopening any search.
4. Only if B2 also fails should B3 be used to separate native-kernel issues from broader donor/contract issues.

## Fallback Ladder

### Level 0 — Normal path
- `WH_KERN_ATTN_MODE=whale_native`
- use blessed `primary @ 1984`

### Level 1 — Kernel-family recovery
- keep `whale_native`
- switch only the env knobs to displaced challenger `fallback @ 928`
- use for poor donor results or obvious instability under the blessed primary kernel

### Level 2 — Diagnostic recovery
- set `WH_KERN_ATTN_MODE=hybrid`
- this activates `whale_fwd_fa3_bwd`
- use only to determine whether the native whale path is the problem

### Level 3 — Contract baseline recovery
- set `WH_KERN_ATTN_MODE=fa3`
- this restores the preserved FA3 path on the same donor contract
- use only as a same-contract anchor baseline, not as a silent permanent fallback

## Prohibited Recovery

Do not recover by:
- reviving `gemini_runs/pr1493_train_gpt.py`
- switching to `scripts/train_gpt.py`
- broad new kernel sweeps
- changing donor geometry and kernel family at the same time

## Logging Requirements

Every run should log:
- donor recipe ID
- trainer contract file
- attention mode
- kernel family
- `DQ`, `DKDV`, `DKDV_MR`, `QMR`
- anchor source evidence path

If those are not printed, the run is not trustworthy enough for comparison.
