# Ablation: BW21_NoisyQAT_9F

Date: 2026-04-06
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Gate (8xGPU, 2000 steps, seed=444)

Status: [ ] pending  [ ] pass  [ ] fail

| Arm | Config | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_ctrl |
|-----|--------|-------------|---------|-------------|---------|-------|---------------|
| BW21-00_ctrl | BWX 9F standard (NOISY_QAT=0) | | | | | | -- |
| BW21-01_nqat | BWX 9F + NOISY_QAT=1 | | | | | | |

Gate target: delta < -0.002 int6_sw_bpb, step_ms within 2ms of control

Notes:

## Full run (8xH100, 600s, seed=444)

Status: [ ] pending  [ ] pass  [ ] fail
(Only if gate passes)

## Confirmation (8xH100, 600s, seed=300)

Status: [ ] pending  [ ] pass  [ ] fail
