# Hypothesis: BW21_NoisyQAT_9F

Date: 2026-04-06
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## What changes (ONE variable only)

Enable Noisy QAT for crawler blocks during training.

Control (NOISY_QAT=0): Standard STE QAT -- all blocks use percentile-clipped
round-to-nearest with straight-through estimator.

Treatment (NOISY_QAT=1): Crawler blocks use noise-injection QAT instead of STE.
Noise is calibrated to the int6 quantization step size (amax/31). Flat blocks
continue using standard STE QAT. No architecture change, no param change.

## Why

The Ouroboros ablation (seed=300, 4xGPU, 600s) showed noisy_qat at -0.00296
int6_sw_bpb versus 9F control. The crawler block's shared weights are quantized
3 times (once per loop), compounding quantization error. Noisy QAT trains the
crawler weights to be robust to this error by injecting calibrated noise during
forward passes, rather than using hard round-to-nearest.

This is the strongest validated single-variable training improvement on the 9F
base. It does not change architecture, model size, or step time.

Note: the Ouroboros ablation ran with loops=2 (config mistake), but the
mechanism is architecture-independent -- noise injection doesn't depend on loop
count.

## Gate target

- int6_sw_bpb delta vs control: < -0.002 (clear signal above noise floor)
- step_avg: within 2ms of control (~110ms on 8xGPU)
- Model params: identical to control (26,270,292)
- Artifact size: within 200KB of control
