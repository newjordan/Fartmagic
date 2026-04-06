# Ouroboros Ablation Sweep — Results

Date: 2026-04-06
Track: crawler
Seed: 300
Config: 4×GPU, 600s wallclock, loop-aware GPTQ, SKIP_EMA=1
Parent: BW5 (9F crawler, loops=3, rope=9,1,1)

## Summary

| Arm | int6_sw_bpb | raw_bpb | steps | step_ms | bytes | delta_vs_ctrl |
|-----|-------------|---------|-------|---------|-------|---------------|
| control | 1.16409381 | 1.1821 | 3225 | 186.09 | 14,535,880 | — |
| noisy_qat | 1.16113228 | 1.1794 | 3380 | 177.53 | 14,769,745 | −0.00296 |
| crawler_int8 | 1.16094168 | 1.1798 | 3377 | 177.70 | 15,548,006 | −0.00315 |
| contractive | 1.16182510 | 1.1798 | 3397 | 176.66 | 14,657,080 | −0.00227 |

## Verdict

All three arms beat control. Ranked by int6_sw_bpb delta:

1. **crawler_int8** (−0.00315): Best quality. But artifact at 15.5MB — only 0.5MB under 16MB cap.
2. **noisy_qat** (−0.00296): Nearly as good, smaller artifact (14.8MB), +155 steps from faster training.
3. **contractive** (−0.00227): Weakest delta but fastest step time (176.7ms) and smallest artifact (14.7MB).

## Observations

- All three arms trained significantly faster than control (176-178ms vs 186ms step_avg).
  Control appears to have had a slower run — this inflates deltas via more training steps.
- Noisy QAT and crawler_int8 are both near −0.003, close to MegaGate noise floor threshold.
  Solid signals but not decisive on a single seed.
- Contractive (SCORE h' = (1-dt)*h + dt*F(h)) is real but modest. Best step time suggests
  it could compound at longer training budgets.
- Crawler_int8 size (15.5MB) is a concern — leaves only 0.5MB headroom to 16MB cap.
  Any architecture expansion (more flat layers, larger model) could bust the limit.

## Next steps

- Seed 444 confirmation needed for all arms to validate signals
- Noisy QAT is the safest bet: good delta + comfortable size budget
- crawler_int8 + noisy_qat combo worth testing if they stack
- Compare these deltas against Helix_ab_3 gate results to decide 4-hour allocation
