# Arch+Sched Sweep — Results

**Date:** TBD
**Pod:** 4×H100
**Seed:** 444

---

## Smoke Test

| step_avg_ms | GPU | NPROC | Status |
|-------------|-----|-------|--------|
| TBD | H100 | 4 | PENDING |

---

## Sweep Results

| case | post_ema_bpb | delta | sliding_bpb | delta | int6_bpb | quant_gap | size_MB | qat_step | steps |
|------|-------------|-------|-------------|-------|----------|-----------|---------|----------|-------|
| baseline | | — | | — | | | | | |
| rope_32 | | | | | | | | | |
| bigram_3072 | | | | | | | | | |
| bigram_4096 | | | | | | | | | |
| qat_early | | | | | | | | | |
| qat_late | | | | | | | | | |
| swa_dense | | | | | | | | | |
| gptq | | | | | | | | | |
| warmdown_4k | | | | | | | | | |

---

## Decision

- [ ] gptq signal validated (biggest expected gain)
- [ ] bigram_3072 signal validated
- [ ] bigram_4096 size gate check
- [ ] rope_32 signal validated
- [ ] qat_early signal validated
- [ ] qat_late signal validated
- [ ] swa_dense signal validated
- [ ] warmdown_4k signal validated

**Outcome:** Pending.
