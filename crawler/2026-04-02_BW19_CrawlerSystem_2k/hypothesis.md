# BW19_CrawlerSystem_2k — Hypothesis

Objective: map crawler interaction dynamics on the 9F platform and maximize crawler leverage without losing submission legality.

Control baseline:
- `NUM_FLAT_LAYERS=9`
- `NUM_CRAWLER_LAYERS=1`, `CRAWLER_LOOPS=3`
- `INST_DIM=32`
- tap off (`CRAWLER_TAP_DIM=0`)
- anchor off (`ANCHOR_DIM=0`)
- smear off (`CRAWLER_LOOP_SMEAR=0`)
- `MODEL_DIM=512`, `MATRIX_LR=0.03`, `EMBED_LR=0.035`
- in-window naive int6 (`SKIP_GPTQ=1`)

Strategy:
1. QUICK stage screens crawler cadence, crawler depth, floor/crawler balance, and mirror-style stabilization interactions.
2. FULL stage replays control + best QUICK arms at 2k for stronger ranking.
3. CALIBRATION stage runs post-window quant policies on best FULL checkpoint.

Important implementation note:
- In the current trainer (`records/.../2026-04-02_Bandit_Wagon_X_9F_8xH100/train_gpt.py`), `DELTA_NET_HEADS` is not consumed.
- “Reverse crawler” in BW19 is therefore tested through mirror/stabilization proxies:
  - `ANCHOR_DIM`
  - `CRAWLER_TAP_DIM` + `CRAWLER_TAP_LOOP_SPECIFIC` + `CRAWLER_TAP_LAYERS`
  - `CRAWLER_LOOP_SMEAR`

## Crawler-System Matrix

WINDOW deltas (must retrain):
- `C00` control
- `C01` cadence extreme low (`CRAWLER_LOOPS=1`)
- `C02` cadence low (`CRAWLER_LOOPS=2`)
- `C03` cadence high (`CRAWLER_LOOPS=4`)
- `C04` crawler depth split (`NUM_CRAWLER_LAYERS=2`, `CRAWLER_LOOPS=2`)
- `C05` crawler depth++ (`NUM_CRAWLER_LAYERS=3`, `CRAWLER_LOOPS=2`)
- `C06` crawler-heavier balance (`NUM_FLAT_LAYERS=7`, `NUM_CRAWLER_LAYERS=2`, `CRAWLER_LOOPS=3`)
- `C07` crawler-dominant balance (`NUM_FLAT_LAYERS=5`, `NUM_CRAWLER_LAYERS=3`, `CRAWLER_LOOPS=3`)
- `C08` pure crawler probe (`NUM_FLAT_LAYERS=0`, `NUM_CRAWLER_LAYERS=4`, `CRAWLER_LOOPS=2`)
- `C09` mirror-lite proxy (`loops=2`, `anchor=16`, `tap=32(shared)`, `smear=1`)
- `C10` mirror-strong proxy (`loops=2`, `anchor=32`, `tap=32(loop-specific)`, `smear=1`)
- `C11` mirror-deep proxy (`loops=3`, `anchor=32`, `tap=32(loop-specific)`, `smear=1`)
- `C12` instruction wide on cadence2 (`INST_DIM=64`)
- `C13` choke residual on cadence2 (`CRAWLER_MLP_CHOKE_DIM=64`, `shape=residual`)
- `C14` QK4 gain on cadence2 (`QK_GAIN_INIT=4.0`)
- `C15` rope retune on cadence2 (`CRAWLER_LOOP_ROPE_SCALES=16,4`)
- `C16` tap deep-only on cadence2 (`CRAWLER_TAP_LAYERS=deep`)
- `C17` tap shallow-only on cadence2 (`CRAWLER_TAP_LAYERS=shallow`)

POST_WINDOW deltas (sequential on best FULL checkpoint):
- `BW19Q-00`: naive int6
- `BW19Q-I8`: naive int6 + crawler int8
- `BW19Q-01`: GPTQ 128x2048
- `BW19Q-01L`: GPTQ 64x1024
- `BW19Q-01H`: GPTQ 256x2048
- `BW19Q-02` (optional): loop-aware GPTQ 128x2048

Primary ranking:
- `int6_sw_bpb` (lower is better)
- plus explicit interaction ratios:
  - `bytes_mb`
  - `size_per_bpb_mb` (size/bpb)
  - `bpb_x_mb`
  - `delta_bytes_vs_control`

Promotion rule:
- Advance only non-dominated arms (quality/size/speed).
