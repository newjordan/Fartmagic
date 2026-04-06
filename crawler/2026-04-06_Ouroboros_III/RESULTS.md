# Results: Ouroboros_III

Date: 2026-04-06
Track: crawler
Parent baseline: BWX 9F (`records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/`)

## Run

Command:
`SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-06_Ouroboros_III/run.sh`

Log:
`/workspace/parameter-golf/crawler/2026-04-06_Ouroboros_III/logs/ouro_iii_s444_20260406_192619.log`

## Metrics (seed=444)

- model_params: `26270292`
- raw_bpb: `1.1561`
- int6_sw_bpb: `1.14461684`
- step_avg_ms: `110.54`
- artifact_bytes: `14721894`
- size_limit: `16777216` (PASS)

## Comparison vs Baseline

- BWX 9F baseline int6_sw_bpb: `1.13867894`
- Ouroboros III int6_sw_bpb: `1.14461684`
- Delta (Ouro III - baseline): `+0.00593790`

## Verdict

`DOES NOT PROMOTE`.

Ouroboros III is under the 16MB cap but regresses quality versus the BWX 9F leader.
This contradicts the isolated single-arm signals (noisy_qat, crawler_int8, contractive)
showing that they did not compose positively in the full stacked production run.
