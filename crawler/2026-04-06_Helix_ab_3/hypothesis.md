# Hypothesis: Helix_ab_3

Date: 2026-04-06
Track: crawler
Parent: legs/2026-03-29_BW5/
Source: parameter-golf-lab/crawler/2026-04-04_Helix_SplitHead/train_gpt.py

## What changes (ONE variable only)

Enable Helix SplitHead dual-stream co-firing architecture, replacing sequential
crawler loops with parallel flat+crawler streams and cross-injection.

Control (HELIX=0): Standard BW5 — 4 flat layers + 1 crawler block x 3 sequential loops.
Treatment (HELIX=1): Helix SplitHead — crawler co-fires alongside each flat layer
with bidirectional linear cross-injection. All crawler attention heads cross-attend
to flat stream (no self-attend). No sequential loops.

Treatment env:
  HELIX=1, HELIX_DIM=384, HELIX_STRIDE=1, HELIX_CROSS_ATTN=0,
  CRAWLER_CROSS_HEADS=8, CRAWLER_LOOPS=1, CRAWLER_LOOP_ROPE_SCALES=9

All other hyperparameters held constant at BW5 defaults, including MUON_WD=0.04.
WD=0.12 (which SplitHead micro found beneficial) is deferred to Helix_ab_4 to
isolate the architecture signal.

## Why

Helix SplitHead showed the largest architectural signal in the crawler research
program at micro scale (dim=256, 5F, 200 steps):

  No helix:           1.9112 BPB
  Standard helix:     1.8605 BPB (-0.051)
  Best SplitHead B6:  1.8333 BPB (-0.078)

Key micro findings:
- Crawler doesn't want self-attention — full cross-attend (cross=4/4) beats every split
- Fat pipe required: dim=192 (~75% of model_dim) optimal
- Stride=1 (frequent firing) wins at fat dims
- Linear projection >> Marco-Polo cross-attention (diverges at width)
- Position-agnostic K (no RoPE on cross-K) for content routing

This gate scales the micro signal to full model (MODEL_DIM=512, 4F) for the first
time. Running on 4xGPU with 2000-step training arms.

## Gate target

- int6_sw_bpb delta vs control: < -0.003 (clear signal above MegaGate noise floor)
- step_avg: < 100ms/step (must not regress catastrophically vs BW5 ~75ms)
- No compilation errors (fullgraph=1 must survive helix code path)

## Risk factors

1. 4F vs 5F: micro tested on 5F; BW5 uses 4F. Fewer flat layers = fewer co-firings.
2. HELIX_DIM=384 at MODEL_DIM=512 is untested — scaling the 75% ratio from micro.
3. CRAWLER_CROSS_HEADS=8 (all heads) with NUM_KV_HEADS=4 (GQA) — verified in code
   but untested at full scale.
4. fullgraph=1 compatibility with helix control flow — loops are fixed-size so
   should unroll at trace time, but first real test.
