# RESULTS — whale_combined_validation_t2048

Heavy confirmation result for `legs/2026-04-17_whale_combined_validation_t2048`: the `10x200` rerun at `(2,2048,8,4,64)` kept whale under FA3 with the stacked DKDV and DQ winners.

## fact
- `logs/combined_validation_20260417_184037.log:3` records the fixed validation shape/backends/rounds/iters, and `:4`, `:5`, `:6` record `dkdv=128,128,8,3`, `dq_tile=128,64,8,5`, and `dq_maxnreg=224`.
- `logs/combined_validation_20260417_184037.log:8` identifies the shape header, `:9` shows whale `fwd=0.068+-0.001ms fwd+bwd=0.316+-0.008ms`, `:10` shows fa3 `fwd=0.079+-0.003ms fwd+bwd=0.319+-0.004ms`, and `:11` ends with `combined validation done`.
- `evidence/combined_validation_20260417_184037.json` records `rounds=10`, `iters_per_round=200`, `result.shape=[2,2048,8,4,64]`, `result.backends.whale_fast.fwd_mean_ms=0.0684275520760566`, `result.backends.whale_fast.fb_mean_ms=0.3158997917920351`, `result.backends.fa3.fwd_mean_ms=0.0793411519639194`, and `result.backends.fa3.fb_mean_ms=0.31939004792273046`.
- The directional stack at T=2048 is now: prior clean pre-sweep baseline `0.3683 ms` in `../2026-04-17_whale_bwd_variant_sweep/RESULTS.md:21`, Leg A winner `0.35600147157907486 ms` in `../2026-04-17_whale_dkdv_wide_autotune_t2048/evidence/dkdv_128_128_8_3_20260417_174933.json` key `result.backends.whale_fast.fb_mean_ms`, Leg B winner `0.3412711681723595 ms` in `../2026-04-17_whale_dq_wide_autotune_t2048/evidence/dq_128_64_8_5_224_20260417_182211.json` key `result.backends.whale_fast.fb_mean_ms`, and combined validation `0.3158997917920351 ms` in `evidence/combined_validation_20260417_184037.json` key `result.backends.whale_fast.fb_mean_ms`.

## inference
- The T=2048 kernel-bench goal is met on the heavier confirmation pass: whale fb `0.315900 ms` is below fa3 `0.319390 ms`, a `-1.09%` crossover at `10x200`.
- The stacked DKDV and DQ winners preserved the crossover under heavier sampling instead of regressing back above FA3.
- Forward remains comfortably ahead too: whale fwd `0.068428 ms` versus fa3 `0.079341 ms` in `evidence/combined_validation_20260417_184037.json` keys `result.backends.whale_fast.fwd_mean_ms` and `result.backends.fa3.fwd_mean_ms`.

## proposal
- Preserve `DKDV=128,128,8,3`, `DQ_TILE=128,64,8,5`, and `DQ_MAXNREG=224` as the T=2048 kernel-bench winner set for swap-in and broader validation. Do not reopen the T=2048 autotune search in this session.

## fact — what is NOT tested here
- This file does not cover D=72, runner swap-in, or training-step wall-time.
- This file does not retest T>=4096 or long-context behavior.
