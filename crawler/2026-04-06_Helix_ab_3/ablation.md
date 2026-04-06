# Ablation: Helix_ab_3

Date: 2026-04-06
Track: crawler
Parent: legs/2026-03-29_BW5/

## Gate (4×GPU, 2000 steps, seed=444)

Status: [ ] pending  [ ] pass  [ ] fail

| Arm | Config | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_ctrl |
|-----|--------|-------------|---------|-------------|---------|-------|---------------|
| HAB3-00_ctrl | BW5 standard (HELIX=0, loops=3) | | | | | | — |
| HAB3-01_helix | SplitHead (dim=384, stride=1, cross=8) | | | | | | |

Gate target: delta < -0.003 int6_sw_bpb, step_ms < 100ms

Notes:

## Full run (8xH100, 600s, seed=444)

Status: [ ] pending  [ ] pass  [ ] fail
(Only if gate passes)

## Confirmation (8xH100, 600s, seed=300)

Status: [ ] pending  [ ] pass  [ ] fail
