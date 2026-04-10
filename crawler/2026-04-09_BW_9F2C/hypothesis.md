# Hypothesis: BW_9F2C
Date: 2026-04-09
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## What changes (ONE variable only)
NUM_CRAWLER_LAYERS: 1 → 2 (default in train_gpt.py)

Everything else identical to BWX 9F parent.

## Why
Corpus ablation A07 showed -0.0119 delta (breakthrough signal) at 4xGPU 1500 steps.
Layer relationship grid confirmed 9F+2C at -0.01478 vs 9F+1C control (2xGPU 1000 steps).
Artifact size in grid: 14.1MB — well under 16MB cap.
Step time: 160ms — fast enough for good step count on 8xH100.

## Gate target
delta vs BWX 9F (1.13868) on int6_sw_bpb. Any improvement = pass.
Artifact must stay under 16MB.
