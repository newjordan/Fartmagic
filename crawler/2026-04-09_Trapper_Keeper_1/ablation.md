# Ablation: Trapper_Keeper_1

Date: 2026-04-09
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Production gate (8xH100, 600s, seed=444)

Status: [ ] pending  [ ] pass  [ ] fail

Command: `SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-09_Trapper_Keeper_1/run.sh`

| Metric | Value |
|--------|-------|
| model_params | |
| raw_bpb | |
| int6_sw_bpb | |
| step_avg_ms | |
| steps | |
| bytes_total | |
| artifact_legal | |

Gate: beat 1.13867894 int6_sw_bpb AND <= 16,000,000 bytes

## Confirmation (8xH100, 600s, seed=300)

Status: [ ] pending

Command: `SEED=300 NPROC_PER_NODE=8 bash crawler/2026-04-09_Trapper_Keeper_1/run.sh`

| Metric | Value |
|--------|-------|
| int6_sw_bpb | |
| bytes_total | |

## Screen evidence

| Source | Config | int6_sw | Environment |
|--------|--------|---------|-------------|
| Grid 2xGPU | 8F+3C | 1.39529 | 2xGPU, 1000 steps |
| Isolated 4xGPU | 8F+3C | 1.34632 | 4xGPU, 1000 steps |
| Grid control | 9F+1C | 1.41256 | 2xGPU, 1000 steps |
