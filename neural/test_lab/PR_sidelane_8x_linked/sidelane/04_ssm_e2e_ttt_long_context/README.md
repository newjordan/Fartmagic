# State-Space Models + E2E TTT + Long Context

## Core Hypothesis

Combining SSM layers with end-to-end test-time training and longer context evaluation can outperform pure attention under strict parameter limits by improving state reuse and adaptation.

## Phase Plan

1. Phase A (proof): hybrid transformer-SSM blocks with optional lightweight TTT adapter updates.
2. Phase B (artifact): constrain TTT state/adapters so export remains within 16MB.
3. Phase C (runtime): benchmark long-context eval schedules and optimize kernels.

## Cross-Repo Tasks

- `parameter-golf-lab`: hybrid block implementation + TTT hooks.
- `sota_crawler`: sequence-length and TTT-step sweeps.
- `ml-lab` / `research_omni`: isolate SSM kernel and recurrence prototypes.

## Initial Ablation Grid

- SSM layer placement: last 2, last 4, alternating
- TTT steps: 0, 1, 2
- Eval context: 2048, 4096, 8192

## Promotion Criteria

- Long-context gains are visible without violating evaluation rules.
- Runtime can be brought toward 10-minute envelope after kernel work.

## Kill Criteria

- Gains appear only at prohibitively expensive context or TTT settings.
