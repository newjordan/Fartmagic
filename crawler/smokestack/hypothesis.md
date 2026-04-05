# Smokestack — Hypothesis

## Concept
Smokestack is a parallel architecture exploration testing whether a flat-dominant,
minimal-recurrence design is more effective than the current crawler recursion model.
This is isolated from the BW-lineage. The crawler recursion line continues separately.

## Parent
BWX 9F: `records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/`
- 1.13867894 int6_sw_bpb, 15,239,617 bytes, 110.19ms/step, 5446 steps (seed 444)

## Thesis
The crawler performs better when focused on flat layers, not "looking inside the crawler."
BWX 9F already embodies this partially (9F, tap-off, no anchor, no smear).
The remaining variable: CRAWLER_LOOPS is still at 3.

## Evidence supporting thesis

| Evidence | Source | Signal |
|----------|--------|--------|
| More flat layers monotonically better | BW11(5F), BW14(6F), BWX(9F) | -0.010, -0.007 |
| Tap-off beats tap-on | BW12 | -0.002 |
| No anchor beats anchor (on tap-off) | BW13 | +0.003 WITH anchor |
| SharedFlat catastrophic | BW7 FLAT-07 | +0.037 |
| Smear null | BW7 SMEAR-01 | -0.00003 |
| Fewer loops better (directional) | BW17 RAPID | -0.054 |
| Crawler recurrence closed as primary path | Research Synthesis | -- |

## ONE variable changed
**CRAWLER_LOOPS**: 3 (control) vs 2 (primary) vs 1 (aggressive)

Everything else locked to BWX 9F defaults:
- NUM_FLAT_LAYERS=9, NUM_CRAWLER_LAYERS=1, INST_DIM=32
- CRAWLER_TAP_DIM=0, ANCHOR_DIM=0, CRAWLER_LOOP_SMEAR=0
- CRAWLER_LOOP_ROPE_SCALES=9,1,1, MODEL_DIM=512
- SKIP_GPTQ=1, WARMDOWN_ITERS=2000, COMPILE_FULLGRAPH=1

## Gate target
delta_vs_control <= -0.003 int6_sw_bpb at 2000 steps

## Risk
- BW17 RAPID signal (-0.054) was on small-token run — absolute values inflated
- loops=1 is untested — could be too aggressive (single pass may not refine enough)
- Rope scales 9,1,1 were tuned for 3 loops — may need adjustment for fewer loops
