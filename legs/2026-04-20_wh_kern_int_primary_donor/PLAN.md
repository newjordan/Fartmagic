# PLAN — 2026-04-20_wh_kern_int_primary_donor

## Goal
Integrate the best whale donor with the blessed 2x whale kernel carry-forward.

Reference planning doc: `docs/wh_kern_int_integration_plan_20260420.md`
Benchmark/fallback contract: `docs/wh_kern_int_benchmark_contract_20260420.md`

## Current anchor
- donor: `whale_l12_dim576_matrixbits5`
- kernel: `primary @ 1984`
- kernel knobs: `DQ=128,64,8,5 / DKDV=128,64,8,3 / DKDV_MR=256`

## Execution plan
1. Run the donor through `pr1493_decompressed.py` contract, not the purged `gemini_runs/pr1493_train_gpt.py` path.
2. Override the attention hook to `custom_whale_attn_fwd`.
3. Feed the blessed kernel env knobs from the current winner anchor.
4. Verify the donor path runs and artifacts export normally.
5. Only reopen kernel choice if integration contradicts the blessed 2x winner.
