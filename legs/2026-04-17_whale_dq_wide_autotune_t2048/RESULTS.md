# RESULTS — whale_dq_wide_autotune_t2048

Sweep result for `legs/2026-04-17_whale_dq_wide_autotune_t2048`: the 14-config DQ sweep at shape `(2,2048,8,4,64)` completed with `DKDV_WINNER=128,128,8,3` locked, and the best sampled DQ setting reduced whale fb again.

## fact
- `logs/sweep_20260417_180550.log:1` shows the 14-config sweep header with dkdv locked at `128,128,8,3`.
- `logs/sweep_20260417_180550.log:29` records the `dq=128,64,8,5 mr=224` launch, `logs/sweep_20260417_180550.log:30` marks it `ok`, and `logs/sweep_20260417_180550.log:31` ends with `[done 20260417_182328]`.
- `logs/dq_128_64_8_5_224_20260417_182211.log:2` identifies the winner shape, and `logs/dq_128_64_8_5_224_20260417_182211.log:3` plus `:4` show whale `fwd=0.075+-0.002ms fwd+bwd=0.341+-0.009ms` versus fa3 `fwd=0.085+-0.001ms fwd+bwd=0.345+-0.003ms`.
- `evidence/dq_128_64_8_5_224_20260417_182211.json` records `rounds=5`, `iters_per_round=100`, `result.shape=[2,2048,8,4,64]`, `result.backends.whale_fast.fwd_mean_ms=0.07542233595997094`, `result.backends.whale_fast.fb_mean_ms=0.3412711681723595`, `result.backends.fa3.fwd_mean_ms=0.08477747201919555`, and `result.backends.fa3.fb_mean_ms=0.3450434566140175`.
- Ranked by the handoff criterion `lowest whale_fast.fb_mean_ms`, the top three sampled DQ settings were: `128,64,8,5@224` at `0.341271` fb vs fa3 `0.345043` (`-1.09%`), `128,64,8,2@224` at `0.351023` vs `0.359937` (`-2.48%`), and `128,64,8,3@160` at `0.352871` vs `0.367839` (`-4.07%`).
- The Leg A winner `128,128,8,3` recorded whale fb `0.35600147157907486` in `../2026-04-17_whale_dkdv_wide_autotune_t2048/evidence/dkdv_128_128_8_3_20260417_174933.json` key `result.backends.whale_fast.fb_mean_ms`.

## inference
- By the explicit handoff criterion, the DQ winner is `DQ_TILE=128,64,8,5` with `DQ_MAXNREG=224` because `0.341271 ms` is the lowest sampled whale fb.
- `128,64,8,3@160` is still signal: it produced the largest relative fb win among the top runs (`-4.07%`), but its absolute whale fb (`0.352871 ms`) is slower than the selected winner's `0.341271 ms`, so it is not the config to carry forward.
- The DQ sweep reduced whale fb from the Leg A winner's `0.356001 ms` to `0.341271 ms`, a `14.73 us` improvement at the same shape.

## proposal
- Lock `DKDV=128,128,8,3`, `DQ_TILE=128,64,8,5`, and `DQ_MAXNREG=224` into the combined validation leg.

## fact — what is NOT tested here
- This file does not claim the global optimum over all possible DQ tiles or `WHALE_BWD_Q_MAXNREG` values; it only reports the 14 sampled configs from this DQ sweep.
- This file does not report the heavy `10x200` confirmation pass or any training-step impact.
