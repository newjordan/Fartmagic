# Hypothesis: crawler_symmetry_ablation

Date: 2026-04-09
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Question

Is the optimal NUM_CRAWLER_LAYERS = CRAWLER_LOOPS?

The layer relationship grid showed 3C is the quality peak at 3 loops, and 4C
reverses. The hypothesis: each crawler layer specializes for one loop iteration.
A 4th crawler layer with only 3 loops has no iteration to attach to — it's
orphaned. If C = loops is a design law, then:

- 4 loops should want 4 crawler layers
- 6 loops should want 6 crawler layers
- 8 loops should want 8 crawler layers

## Arms

Flat layers held constant at 8F (best from grid data so far). Only
CRAWLER_LOOPS and NUM_CRAWLER_LAYERS change. ROPE_SCALES uses naive
pattern (9 followed by 1s) to isolate the symmetry variable.

| Arm | Flat | Crawler | Loops | ROPE_SCALES | Effective depth |
|-----|------|---------|-------|-------------|-----------------|
| A0 | 8 | 3 | 3 | 9,1,1 | 8 + 3×3 = 17 (control) |
| A1 | 8 | 4 | 4 | 9,1,1,1 | 8 + 4×4 = 24 |
| A2 | 8 | 6 | 6 | 9,1,1,1,1,1 | 8 + 6×6 = 44 |
| A3 | 8 | 8 | 8 | 9,1,1,1,1,1,1,1 | 8 + 8×8 = 72 |

## Environment

- 4xGPU, 1000 steps, seed=444
- BWX 9F production train_gpt.py — zero diff except the three knobs
- TRAIN_BATCH_TOKENS=393216 (standard 4x batch)

## Risk

- A2 and A3 are very deep (44 and 72 effective passes). Step time will be
  high. May OOM on large configs. That's OK — if it crashes, we know the
  practical ceiling.
- Size will be over 16MB on A2/A3. This test is about the quality signal,
  not production legality. Size optimization comes later.

## What this tells us

- If A1 (4C+4L) beats A0 (3C+3L): symmetry hypothesis confirmed at order 4
- If each arm beats the previous: C=loops is a scaling law
- If there's a peak: we know the practical ceiling for symmetric depth
- Step_ms per arm tells us the production wallclock tradeoff
