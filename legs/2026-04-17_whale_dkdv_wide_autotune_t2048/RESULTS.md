Sweep result for `legs/2026-04-17_whale_dkdv_wide_autotune_t2048`: the 18-config DKDV sweep at shape `(2,2048,8,4,64)` completed, and the best sampled `WHALE_BWD_KV_TMA_CONFIG` flipped the T=2048 headline versus the prior clean baseline.

## fact
- `logs/sweep_20260417_172853.log:1` shows the sweep header for 18 configs at shape `(2,2048,8,4,64)`.
- `logs/sweep_20260417_172853.log:25` records the `128,128,8,3` launch.
- `logs/sweep_20260417_172853.log:31` records the `256,64,8,3` launch, and `logs/sweep_20260417_172853.log:32` says that config failed.
- `logs/sweep_20260417_172853.log:39` ends with `[done 20260417_180206]`.
- `logs/dkdv_256_64_8_3_20260417_175513.log:46` reports the failure cause: `out of resource: shared memory, Required: 280640, Hardware limit: 232448`.
- `logs/dkdv_128_128_8_3_20260417_174933.log:2-4` shows whale `fwd=0.076+-0.003ms fwd+bwd=0.356+-0.004ms` and fa3 `fwd=0.091+-0.004ms fwd+bwd=0.368+-0.015ms`.
- `logs/dkdv_128_64_8_3_20260417_174012.log:2-4` shows whale `fwd=0.077+-0.003ms fwd+bwd=0.361+-0.007ms` and fa3 `fwd=0.089+-0.002ms fwd+bwd=0.366+-0.005ms`.
- `evidence/dkdv_128_128_8_3_20260417_174933.json` records `rounds=5`, `iters_per_round=100`, `result.shape=[2,2048,8,4,64]`, `result.backends.whale_fast.fwd_mean_ms=0.07649343993514776`, `result.backends.whale_fast.fb_mean_ms=0.35600147157907486`, `result.backends.fa3.fwd_mean_ms=0.09060588802397251`, and `result.backends.fa3.fb_mean_ms=0.36782969582080843`.
- `evidence/dkdv_128_64_8_3_20260417_174012.json` records `result.backends.whale_fast.fwd_mean_ms=0.07722035207599401`, `result.backends.whale_fast.fb_mean_ms=0.36146611189842226`, `result.backends.fa3.fwd_mean_ms=0.08898111997544765`, and `result.backends.fa3.fb_mean_ms=0.36573875206708906`.
- `evidence/dkdv_128_64_4_2_20260417_173626.json` records `result.backends.whale_fast.fwd_mean_ms=0.078104`, `result.backends.whale_fast.fb_mean_ms=0.367765`, `result.backends.fa3.fwd_mean_ms=0.085704`, and `result.backends.fa3.fb_mean_ms=0.357474`.
- Ranked across successful JSONs: winner `128,128,8,3` at `0.356001` fb vs fa3 `0.367830` (`-3.22%`), runner-up `128,64,8,3` at `0.361466` vs `0.365739` (`-1.17%`), third `128,64,4,2` at `0.367765` vs `0.357474` (`+2.88%`).
- The prior clean baseline in `legs/2026-04-17_whale_bwd_variant_sweep/RESULTS.md:21` reported `bwd_fused_delta_tma` at T=2048 with whale fb `0.3683 ms` vs fa3 `0.3555 ms` (`+3.6%`).
- The leg hypothesis states that only `WHALE_BWD_KV_TMA_CONFIG` varies at this shape in `hypothesis.md:9` and `hypothesis.md:15`.

## inference
- This DKDV-only sweep is enough to change the T=2048 headline for the sampled winner: the prior clean baseline was `+3.6%` slower, while `128,128,8,3` is `-3.22%` faster on fb at the same shape.
- The shared-memory failure on `256,64,8,3` narrows the usable config space for this shape; it is not a viable winner candidate for the running sweep.

## proposal
- Lock `DKDV_WINNER=128,128,8,3` into the running DQ sweep.

## fact — what is NOT tested here
- This file does not claim the global optimum over all possible `WHALE_BWD_KV_TMA_CONFIG` values; it only reports the 18 sampled configs from this DKDV sweep.
- This file does not test the DQ sweep result, the combined validation leg, or any training-end-to-end impact.
