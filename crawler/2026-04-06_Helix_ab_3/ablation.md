# Ablation: Helix_ab_3

Date: 2026-04-06
Track: crawler
Parent: legs/2026-03-29_BW5/

## Gate (2000 steps, seed=444)

Status: [ ] pending  [ ] pass  [x] fail

| Arm | Config | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_ctrl |
|-----|--------|-------------|---------|-------------|---------|-------|---------------|
| HAB3-00_ctrl | BW5 standard (HELIX=0, loops=3) | 14,462,508 | 1.3080 | 1.28905275 | 72.77 | 8,887,004 | — |
| HAB3-01_helix | SplitHead (dim=384, stride=1, cross=8) | 15,183,916 | 1.3351 | 1.42860043 | 86.35 | 9,320,278 | +0.13954768 |

Gate target: delta < -0.003 int6_sw_bpb, step_ms < 100ms

Notes:
- Run command used: `SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-06_Helix_ab_3/gate.sh`
- Summary file: `/workspace/parameter-golf/crawler/2026-04-06_Helix_ab_3/results/summary_s444_20260406_194948.tsv`
- Result: hard fail on quality (+0.13954768 vs control). Do not promote.

## Full run (8xH100, 600s, seed=444)

Status: [ ] pending  [ ] pass  [x] fail
(Only if gate passes)

## Confirmation (8xH100, 600s, seed=300)

Status: [ ] pending  [ ] pass  [ ] fail
