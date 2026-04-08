# BW18_9F_DeltaMatrix_2k — Hypothesis

Objective: lock the 9F floor and run a complete, efficient delta matrix over all currently actionable crawler knobs.

Stable base (control):
- `NUM_FLAT_LAYERS=9`
- `NUM_CRAWLER_LAYERS=1`, `CRAWLER_LOOPS=3`, `INST_DIM=32`
- tap-off (`CRAWLER_TAP_DIM=0`), no anchor (`ANCHOR_DIM=0`)
- `MODEL_DIM=512`, `MATRIX_LR=0.03`, `EMBED_LR=0.035`
- naive int6 in-window (`SKIP_GPTQ=1`)

Strategy:
1. QUICK stage screens broad deltas on the 9F platform.
2. FULL stage replays only control + top QUICK candidates at 2k.
3. CALIBRATION stage runs sequential post-window quant/calibration on best FULL checkpoint.

Classification:
- Must retrain (WINDOW): architecture/cadence/width/optimizer interaction deltas.
- Sequential after one window (POST_WINDOW): quantization and calibration deltas with `SKIP_TRAIN=1`.

## Delta Matrix

WINDOW deltas (must retrain):
- `D00` control 9F
- `D01` loops=2
- `D02` loops=4
- `D03` crawler layers=2, loops=2
- `D06` width 576
- `D07` width 640
- `D24` `QK_GAIN_INIT=4.0` (QK4)
- `D08` tap shared dim=32
- `D09` tap loop-specific dim=32
- `D10` anchor dim=16
- `D11` anchor dim=32
- `D14` choke flat-64
- `D15` choke residual-64
- `D16` crawler MLP mult=5.0
- `D17` crawler MLP mult=7.0
- `D18` delta-net heads=2
- `D23` flat weight share=1 (expected high risk)
- `D04` inst dim=16
- `D05` inst dim=64
- `D12` loop smear=1
- `D13` rope scales=16,4,1
- `D19` XSA last N=9
- `D20` matrix lr=0.028
- `D21` matrix lr=0.032
- `D22` embed lr=0.030

POST_WINDOW deltas (sequential on best FULL checkpoint):
- `BW18Q-00`: naive int6
- `BW18Q-I8`: naive int6 + crawler int8
- `BW18Q-01`: GPTQ 128x2048
- `BW18Q-01L`: GPTQ 64x1024
- `BW18Q-01H`: GPTQ 256x2048
- `BW18Q-02` (optional): loop-aware GPTQ 128x2048

Primary decision criteria:
- Rank by `int6_sw_bpb` vs control.
- Reject unstable arms (non-zero exit).
- Keep size/speed visible (`bytes`, `step_ms`) while optimizing quality.

## Queued Next (After BW18 Completes)

Concept: intertwined crawler/correction mechanism ("gear train") to replace part of flat-floor function with recurrent error-correction passes.

Queued experiment family:
- Pure crawler floor stress: `NUM_FLAT_LAYERS=0`, varying `NUM_CRAWLER_LAYERS` and `CRAWLER_LOOPS`.
- Inverse-style correction passes via `DELTA_NET_HEADS` on top of pure crawler floor.
- State-handoff variants (anchor/tap) to stabilize pass-to-pass refinement.
- Stability sweeps (loop counts, instruction width, scaling) to prevent drift/echo.

Execution policy:
- Run only after current BW18 queue is complete.
- Same split: WINDOW retrain arms first, then POST_WINDOW quant on best checkpoint.
