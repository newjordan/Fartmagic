# Hypothesis — whale_dkdv_wide_autotune_t2048

`fact`: Profile leg 2026-04-17_whale_bwd_delta_tma_profile attributes the whale↔FA3 bwd gap to `_attn_bwd_dkdv_inline_delta_tma_kernel` at 105.79 µs (largest single kernel in the whole pipeline; FA3 entire fused bwd is 51.16 µs).

`fact`: Existing `_bwd_kv_inline_tma_configs()` (vault/whale_kernel_triton.py:159-174) autotunes over BM×BN ∈ {(64,64),(64,128),(128,64),(128,128)} × warps ∈ {4,8} × stages ∈ {3,4}, maxnreg=224 fixed. Current winner at D=64 T=2048 reported by prior bench: BM=128, BN=64, warps=8, stages=2.

`hypothesis`: Wider autotune grid — specifically BM=256 (new), BN=256 (new), and stages=2 additions — may find a better dkdv config at T=2048.

`proposal`: Sweep 18 single-config variants via `WHALE_BWD_KV_TMA_CONFIG` env override. Single-GPU, serial. Each config runs the existing `bench_stable.py` at shape (2,2048,8,4,64), backends=whale_fast,fa3, 5 rounds × 100 iters bf16 causal. Output JSON per config. Rank by `whale_fast.fb_mean_ms`.

## Scope (one variable per run — the dkdv config)
- Shape locked: (2, 2048, 8, 4, 64), causal=True, bf16, H=8, KV=4, D=64
- WHALE_BWD_VARIANT=fused_delta_tma (fixed across runs)
- WHALE_FUSED_DELTA_T_MAX=3072 (fixed)
- Only `WHALE_BWD_KV_TMA_CONFIG` varies
- maxnreg=224 (locked at autotune default for this sweep; a second leg can sweep maxnreg)

## Success criteria
- 18 config runs complete, one JSON each
- Best config's `whale_fast.fb_mean_ms` recorded
- Improvement vs current-best (fused_delta_tma baseline of 0.3683ms fb) quantified

## Config grid (BM,BN,warps,stages)

```
64,64,4,2    64,64,8,2    64,128,4,2   64,128,8,2
128,64,4,2   128,64,8,2   128,64,8,3   128,64,8,4
128,128,4,2  128,128,8,2  128,128,4,3  128,128,8,3
256,64,4,2   256,64,8,2   256,64,8,3   256,128,8,2
64,256,4,2   64,256,8,2
```

## Risks
- Some configs may fail Triton compile on 3.6 (pow-2 BM constraint for tl.arange). BM=256 is pow-2 so should compile. BN=256 is pow-2 so should compile. Log any compile failures as distinct from perf results.
- Warmup of 50 iters may be thin for autotune-free (single-config) runs. bench_stable.py uses 50-iter thorough warmup which is sufficient for a single config.
