# Ablation: BW22_LoopDepth_9F

Date: 2026-04-07
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Gate (8xGPU, 2000 steps, seed=444)

Status: [ ] pending  [ ] pass  [ ] fail

| Arm | LOOPS | ROPE_SCALES | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_ctrl |
|-----|-------|-------------|-------------|---------|-------------|---------|-------|---------------|
| A0_ctrl | 3 | 9,1,1 | | | | | | -- |
| A1_loop4_naive | 4 | 9,1,1,1 | | | | | | |
| A2_loop4_battery | 4 | 9,3,1,1 | | | | | | |
| A3_loop5_battery | 5 | 9,3,1,1,1 | | | | | | |
| A4_loop5_prog | 5 | 9,5,3,1,1 | | | | | | |

Gate target: any negative delta justifies exploration. Step_ms critical for 4h budget calc.

Notes:

## 4-hour production run

Status: [ ] pending
(Config chosen based on gate results)
