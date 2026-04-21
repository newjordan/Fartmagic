# PLAN — 2026-04-20_wh_kern_int_primary_donor

## Goal
Integrate the best whale donor with the blessed 2x whale kernel carry-forward.

Reference planning doc: `docs/wh_kern_int_integration_plan_20260420.md`
Benchmark/fallback contract: `docs/wh_kern_int_benchmark_contract_20260420.md`

## Current anchor
- donor: `whale_l12_dim576_matrixbits5`
- kernel: `primary @ 1984`
- kernel knobs: `DQ=128,64,8,5 / DKDV=128,64,8,3 / DKDV_MR=256`

## Executable benchmark ladder
- `B0`: same-contract FA3 baseline
- `B1`: blessed whale native integration
- `B2`: fallback kernel recovery
- `B3`: hybrid diagnostic only

Selection is through `WH_KERN_BENCHMARK_ID`, not manual whale env edits.

## Execution plan
1. Run the donor through `pr1493_decompressed.py` contract, not the purged `gemini_runs/pr1493_train_gpt.py` path.
2. Resolve B0/B1/B2/B3 through `tracked_env.sh` and `benchmark_matrix.json`.
3. Override the attention hook to `custom_whale_attn_fwd` for B1/B2, and `whale_fwd_fa3_bwd` only for B3.
4. Feed the blessed kernel env knobs from the current winner anchor, or the displaced fallback anchor for B2.
5. Verify the donor path runs and artifacts export normally.
6. Only reopen kernel choice if B1 fails and B2 cannot recover.
