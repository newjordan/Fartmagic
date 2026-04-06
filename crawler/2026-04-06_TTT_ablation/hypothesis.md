# E2E TTT Ablation — Both Architecture Arms

Date: 2026-04-06
Track: crawler
Source: arXiv 2512.23675 (TTT-E2E)

## What changes

Add E2E Test-Time Training layer to crawler blocks. Small fast network (ttt_dim × ttt_dim)
adapts per-window via gradient step during both training (E2E) and eval.

Inner objective: predict next-position features in TTT space (causal self-supervised).
Fast weights are learned to be good starting points for adaptation.
After adaptation, output is added as residual to the crawler stream.

## Test matrix (4 arms)

| Arm | Architecture | TTT | Config |
|-----|-------------|-----|--------|
| TTT-00 | Ouroboros (HELIX=0, loops=3) | OFF | Control |
| TTT-01 | Ouroboros (HELIX=0, loops=3) | ON  | TTT_DIM=32 |
| TTT-02 | Helix SplitHead (HELIX=1, cross=8) | OFF | Control |
| TTT-03 | Helix SplitHead (HELIX=1, cross=8) | ON  | TTT_DIM=32 |

## Why

The crawler's recurrence is primitive test-time computation (same weights, multiple passes).
E2E TTT makes the refinement input-adaptive. Battery (9,1,1) is static per-loop specialization;
TTT is the learned version. Competition eval window is 10 minutes — plenty of budget for TTT
gradient steps (~440s available after ~160s base eval).

Organizers specifically requested this test.

## Gate target

- Delta better than −0.003 int6_sw_bpb for either architecture
- Step time overhead < 30% (TTT adds autograd.grad in forward, fullgraph disabled)
- No NaN/divergence from inner loop
