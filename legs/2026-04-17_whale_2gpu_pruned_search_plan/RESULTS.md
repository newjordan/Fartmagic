# RESULTS — whale_2gpu_pruned_search_plan

Benched 2026-04-18 on Vast instance `35165421` (2x H100 SXM, torch `2.11.0+cu130`, CUDA `13.0`, FA3 wheel cu130). Wave 1 used `5x100` DQ screening. Wave 2 used `10x200` heavy validation plus a `5x100` DKDV maxnreg follow-up. Raw JSONs are committed in `evidence/`; promotion metadata is in `logs/wave2_select_20260418_010948.log`.

## fact — wave 1 selected `DQ_TILE=128,64,8,2`

- `logs/wave2_select_20260418_010948.log` selected `worker0_dqtile_128_64_8_2_dqmr_160_dkdvmr_256_20260418_010329.json` for promotion.
- `evidence/worker0_dqtile_128_64_8_2_dqmr_160_dkdvmr_256_20260418_010329.json: results[0].backends.whale_fast.fb_mean_ms = 0.29804377633333207`
- `evidence/worker0_dqtile_128_64_8_2_dqmr_160_dkdvmr_256_20260418_010329.json: results[0].backends.fa3.fb_mean_ms = 0.3122028158903122`
- `evidence/worker1_dqtile_128_64_8_5_dqmr_224_dkdvmr_256_20260418_010404.json: results[0].backends.whale_fast.fb_mean_ms = 0.30412076783180236`
- `evidence/worker1_dqtile_128_64_8_5_dqmr_224_dkdvmr_256_20260418_010404.json: results[0].backends.fa3.fb_mean_ms = 0.32891123127937316`

## fact — DKDV maxnreg ridge collapses to `256`

| DKDV maxnreg | whale fb (ms) | fa3 fb (ms) | Δ ms vs FA3 | evidence |
|--------------|---------------|-------------|-------------|----------|
| 160 | 0.356277056456 | 0.338636159837 | +0.017640896618 | `evidence/wave2_dkdvmr_160_dq_128_64_8_2_dqmr_160_20260418_011418.json` |
| 192 | 0.507277185619 | 0.336664703786 | +0.170612481833 | `evidence/wave2_dkdvmr_192_dq_128_64_8_2_dqmr_160_20260418_011535.json` |
| 224 | 0.328575679839 | 0.319050431788 | +0.009525248051 | `evidence/wave2_dkdvmr_224_dq_128_64_8_2_dqmr_160_20260418_011652.json` |
| 256 | 0.304314432144 | 0.313158591866 | -0.008844159722 | `evidence/wave2_dkdvmr_256_dq_128_64_8_2_dqmr_160_20260418_011808.json` |

## fact — heavy validations are now green across multiple DQ points

| DQ tile | Q maxnreg | whale fwd (ms) | whale fb (ms) | fa3 fwd (ms) | fa3 fb (ms) | Δ ms vs FA3 | evidence |
|---------|-----------|----------------|---------------|--------------|-------------|-------------|----------|
| `128,64,8,2` | 160 | 0.070261680095 | 0.299601856083 | 0.088630255952 | 0.317981455892 | -0.018379599810 | `evidence/wave2_validate_dq_128_64_8_2_dqmr_160_dkdvmr_256_20260418_011418.json` |
| `128,64,8,2` | 192 | 0.070443216056 | 0.301266575933 | 0.085730624035 | 0.317939296082 | -0.016672720149 | `evidence/manual_validate_dq_128_64_8_2_dqmr_192_dkdvmr_256_20260418_011646.json` |
| `128,64,8,2` | 224 | 0.072631728012 | 0.298958048120 | 0.086060463995 | 0.316709568158 | -0.017751520038 | `evidence/manual_validate_dq_128_64_8_2_dqmr_224_dkdvmr_256_20260418_011900.json` |
| `128,64,8,2` | 256 | 0.069578400085 | 0.298562224105 | 0.085185455974 | 0.318293856025 | -0.019731631920 | `evidence/manual_validate_dq_128_64_8_2_dqmr_256_dkdvmr_256_20260418_012119.json` |
| `128,64,8,5` | 224 | 0.069888336012 | 0.299288496166 | 0.086008688033 | 0.343610080525 | -0.044321584359 | `evidence/manual_validate_dq_128_64_8_5_dqmr_224_dkdvmr_256_20260418_012031.json` |

## fact — heavy replicate spread is real, but still green

- `evidence/replicate_validate_dq_128_64_8_2_dqmr_224_dkdvmr_256_20260418_012226.json: results[0].backends.whale_fast.fb_mean_ms = 0.318370576218`, `results[0].backends.fa3.fb_mean_ms = 0.359715664282`
- `evidence/replicate_validate_dq_128_64_8_2_dqmr_256_dkdvmr_256_20260418_012338.json: results[0].backends.whale_fast.fb_mean_ms = 0.301880624130`, `results[0].backends.fa3.fb_mean_ms = 0.307286895916`

## inference — breakthrough and current anchor

- The whale kernel is now under FA3 on heavy `10x200` runs at the exact competition shape `(2,2048,8,4,64)` with scratch-controlled source commit `16cf4c5`.
- `DKDV_MAXNREG=256` is the only viable register cap in this local basin. Lower DKDV caps re-open the loss immediately.
- The primary winning family is `DQ_TILE=128,64,8,2`; within the tested range the DQ maxnreg ridge improved from `160` to `224` to `256`.
- `DQ_TILE=128,64,8,5` at `Q_MAXNREG=224` is a real fallback family, not a fluke one-off.
- Replicate spread is still wide enough that the exact ranking between nearby green points needs more repeatability work, even though the whale `<` FA3 crossover itself is now established.

## proposal — next experiment

1. Preserve `DQ_TILE=128,64,8,2`, `Q_MAXNREG=256`, `DKDV_TILE=128,128,8,3`, `DKDV_MAXNREG=256` as the new anchor.
2. Continue only the DQ high-side ridge above the original grid (`Q_MAXNREG=288`, `320`, ...) with DKDV fixed at `256`.
3. Run a dedicated repeatability leg on `128,64,8,2@224` vs `128,64,8,2@256` with repeated heavy confirmations to lock the exact winner.
