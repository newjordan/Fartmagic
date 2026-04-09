# Hypothesis: crawler_layer_relationship_ablation

Date: 2026-04-09
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Question

What is the dynamic relationship between flat layers and crawler layers?

corpus_ablations_v1 showed NUM_CRAWLER_LAYERS=2 is -0.0119 on the 9F base — the
biggest single-variable delta in the crawler campaign. But we don't know if that
signal depends on having 9 flat layers feeding it. Would 4 crawlers on 6F be
even better? Would 3 crawlers on 7F? We have no data on the interaction surface.

## Experiment

5x4 grid sweep of NUM_FLAT_LAYERS (5-9) x NUM_CRAWLER_LAYERS (1-4).
20 cells. Every cell uses identical BWX 9F production config except the two
knobs being swept. Zero other changes.

This is a big-fat-signal screen. We are looking for macro-level shape of the
quality surface, not marginal effects. Fine-tuning comes after we know the shape.

## Grid

```
         1C      2C      3C      4C
  9F   [control]  [ ]     [ ]     [ ]
  8F     [ ]      [ ]     [ ]     [ ]
  7F     [ ]      [ ]     [ ]     [ ]
  6F     [ ]      [ ]     [ ]     [ ]
  5F     [ ]      [ ]     [ ]     [ ]
```

## Environment

- 2xGPU, 1000 steps, seed=444
- TRAIN_BATCH_TOKENS=196608 (half batch for 2xGPU)
- CRAWLER_LOOPS=3 for all arms (held constant)
- CRAWLER_LOOP_ROPE_SCALES=9,1,1 for all arms (held constant)
- All other env vars identical to BWX 9F production run.sh

## Metric

Raw training BPB at stop step. No post-train eval sweeps. No quant arms.
The script naturally reports int6_sw_bpb too — we capture it but this screen
is about the training signal, not the quant path.

## What this tells us

- Whether crawler layers substitute for flat layers or depend on them
- Where the quality surface peaks
- Where artifact size / step time makes a config impractical for 600s production
- Which region of the grid to zoom into for proper 2000-step 8x gates
