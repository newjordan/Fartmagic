# BW16_WidthSweep_2k — Hypothesis

Goal: measure width sensitivity (and delta) on the currently best depth regime.

## Setup

- Fix architecture to tap-off Nightcrawler with `NUM_FLAT_LAYERS=6`.
- Sweep `MODEL_DIM` only.
- 2k-step gate on 4x GPUs for quick signal.

## Primary hypothesis

- There is additional gain beyond 6F by increasing width from 512 to a larger value.

## Secondary hypothesis

- There is a width knee where quality gains flatten while latency and bytes continue increasing.

## Promotion rule

- Promote if `delta_vs_control <= -0.0030` and artifact size/step time stay operationally acceptable.
