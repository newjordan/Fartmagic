# Whale T2048 Breakthrough — 2026-04-18

Whale is now under FA3 on heavy validation at the competition kernel-bench shape `(2,2048,8,4,64)`.

## Current best committed point

- `DQ_TILE=128,64,8,2`
- `Q_MAXNREG=256`
- `DKDV_TILE=128,128,8,3`
- `DKDV_MAXNREG=256`
- `evidence/manual_validate_dq_128_64_8_2_dqmr_256_dkdvmr_256_20260418_012119.json: results[0].backends.whale_fast.fb_mean_ms = 0.2985622241050005`
- `evidence/manual_validate_dq_128_64_8_2_dqmr_256_dkdvmr_256_20260418_012119.json: results[0].backends.fa3.fb_mean_ms = 0.31829385602474214`

## Why this counts as a breakthrough

- The crossover is no longer a single light-pass tease; heavy `10x200` validations stayed green at `Q_MAXNREG in {160,192,224,256}` on the `128,64,8,2` family.
- `DKDV_MAXNREG=256` survived the follow-up ridge sweep while `160`, `192`, and `224` all lost, so the backward register-cap problem has a clear local answer.
- A second DQ family, `128,64,8,5@224`, also stayed green, so the win is not confined to a single brittle point.

## Caution

- Replicate spread is still material. The committed evidence establishes whale `<` FA3 at T=2048 on this stack, but the exact ranking between `Q_MAXNREG=224` and `256` still deserves repeatability work.

## Immediate next move

- Treat `128,64,8,2 / 256 / 128,128,8,3 / 256` as the new anchor.
- Spend future search budget only on `Q_MAXNREG > 256` and on repeatability checks of `224` vs `256`.
