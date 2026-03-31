# Arch+Sched Sweep — Hypothesis

**Date:** 2026-03-31
**Parent:** Rascal II (1.10986874 BPB, seed 444)
**Pod:** 4×H100
**Seed:** 444

---

## What this sweep is

Six 1-variable probes against the Rascal II baseline. All run at
`MAX_WALLCLOCK_SECONDS=600`, `NPROC=4`. On 4×GPU the LR warmdown is active
from step 1 (warmdown_ms ≈ 637s > 600s wallclock), so QAT fires at ~step 2800
and SWA at ~step 2650 — both inside the window.

---

## Cases

### baseline
Exact `sota_now.sh` env. Control. Expected: 1.10986874 BPB (may vary slightly
from wallclock jitter on 4×GPU vs 8×GPU).

### rope_32
`ROPE_DIMS`: 16 → 32
**Hypothesis:** More rotary dimensions give the model richer positional
encoding. Locked at 16 for conservatism; 32 may help without hitting the size
gate (purely algorithmic, zero size impact).

### bigram_4096
`BIGRAM_VOCAB_SIZE`: 2048 → 4096
**Hypothesis:** Larger bigram table captures more 2-gram statistics at training
time. **Risk:** size gate — bigram table adds ~2×params to the embedding head.
Watch `size_bytes` vs 16MB.

### qat_early
`LATE_QAT_THRESHOLD`: 0.15 → 0.25
**Hypothesis:** Starting QAT earlier (~step 2420) gives more quantization-aware
fine-tuning steps before the run ends. Could tighten quant_gap.

### qat_late
`LATE_QAT_THRESHOLD`: 0.15 → 0.05
**Hypothesis:** Starting QAT later (~step 3120) lets the float model converge
further before QAT noise is introduced. Could improve post_ema_bpb at the cost
of fewer QAT steps.

### swa_dense
`SWA_EVERY`: 50 → 10
**Hypothesis:** More frequent weight averaging produces a smoother ensemble.
SWA fires at the same step, but accumulates 5× more snapshots before the run
ends. May help sliding_bpb without touching the training dynamics.

---

## What to look for

| Metric | Why |
|--------|-----|
| `sliding_bpb` | Race metric — this is the score |
| `post_ema_bpb` | Float model quality; isolates training signal from quant |
| `quant_gap` | `int6_bpb - post_ema_bpb`; lower = QAT working |
| `size_bytes` | Must stay ≤ 16,000,000 bytes |
| `qat_step` | Confirms threshold fired at expected step |

A case is interesting if `sliding_bpb` drops vs baseline. `post_ema_bpb`
dropping but `sliding_bpb` flat = quant degradation eating the gain.
