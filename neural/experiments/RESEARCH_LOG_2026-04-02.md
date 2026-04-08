# Research Log — 2026-04-02

## Anchor
- Safe legal SOTA: `1.10986874` BPB (seed 444), 15,554,053 bytes
- Source: `vault/train_gpt_rascal_sota_REAL.py`
- SHA256: `0ec1f462ab39fd601b18f2b086f6283a0c8db3d2a9780a92dfb206ec46e067cb`

## Competition snapshot (as of 2026-04-02)
- Merged board top: 1.1147 (PR #1019)
- Live open frontier: 1.0962 (PR #1105), 1.09785 (PR #1218)
- Our gap to frontier: ~0.0137 BPB
- PR #1176 claims 1.0914 with SLOT (contested legality)

---

## Experiment 1: 4xGPU 2k Proxy Build Sweep

**Status: CLEAN DATA**

- Seed: 444
- GPUs: 4
- Iterations: 2000
- Train shards: 16 (reduced for disk)
- WARMDOWN_ITERS: 0
- SKIP_FINAL_EVAL: 1
- POST_EMA_DIAGNOSTIC: 1
- Runner: `experiments/rascal_hunt_2k/run_signal_hunt_4gpu.sh build`

| case | post_ema_bpb | roundtrip_bpb | model_bytes | total_bytes |
|---|---|---|---|---|
| ctrl | 1.2241 | 1.22930326 | 16,164,397 | 16,282,915 |
| qkgain4 | 1.2219 | 1.22718580 | 15,112,836 | 15,231,354 |
| bigram2816 | 1.2225 | 1.22734626 | 16,279,975 | 16,398,493 |
| gptq | 1.2225 | 1.22753185 | 15,869,962 | 15,988,480 |
| qk4_gptq | 1.2220 | 1.22701091 | 15,771,144 | 15,889,662 |
| qk4_bigram2816 | 1.2221 | 1.22689641 | 15,958,980 | 16,077,498 |

### Interpretation
- QK4 is the cleanest single-variable proxy win: best BPB AND biggest size savings (~1.05MB)
- qk4_gptq is the best balanced combo
- bigram2816 alone increases size too much
- All combos with QK4 compress better than ctrl

### CAUTION
- This is 2k steps on 4 GPUs with 16 shards and WARMDOWN=0
- Proxy results did NOT transfer to full runs (see Experiment 3 below)
- Proxy model sizes are NOT predictive of full-run model sizes

---

## Experiment 2: 64-Step QK+SLOT Size Smoke

**Status: SIZE DATA ONLY (early regime, not predictive)**

- Seed: 300
- GPUs: 4
- Iterations: 64
- EXIT_AFTER_SIZE_ONLY: 1
- Runner: `experiments/slot_fix_spark/run_qk_slot_hunt.sh short`

| case | model_bytes | total_bytes |
|---|---|---|
| ctrl | 4,513,230 | 4,635,554 |
| qk4 | 4,211,127 | 4,333,451 |
| qk4_slot_p30 | 4,203,024 | 4,325,348 |
| qk4_slot_p30_s2 | 4,207,710 | 4,330,034 |

### Interpretation
- SLOT is size-neutral in early regime
- But 64 steps is too weak to validate for full runs
- Short harness only answers: "does SLOT cause immediate serialization bloat?" (no)
- Does NOT answer: "does SLOT cause full-run compression blowup?" (yes, it does)

---

## Experiment 3: Lucky II Full Run (QK4 + SLOT)

**Status: CLEAN RUN, BAD RESULT**

- Seed: 300
- GPUs: 8xH100
- Wallclock: 600s
- Steps: 6627
- File: `neural/experiments/Lucky II/modify_me.py`
- Changes vs vault: QK_GAIN_INIT=4.0, SLOT_ENABLED=1, legal post-export SLOT eval helper

| metric | value |
|---|---|
| post_ema_bpb | 1.1331 |
| model int6+zstd | 16,633,225 bytes |
| total int6+zstd | 16,757,840 bytes |
| roundtrip_bpb | 1.14326440 |
| sliding_bpb | 1.11386773 |

### Verdict: FAILED
- 16.6MB is way over budget
- Sliding BPB 1.1139 is worse than safepoint 1.1099
- SLOT integration into main trainer still causes size blowup at full scale

---

## Experiment 4: QK4 Contender Full Run (no SLOT)

**Status: POLLUTED — NOT A CLEAN A/B**

- Seed: 300
- GPUs: 8xH100
- Wallclock: 600s
- Steps: 6626
- File: `neural/experiments/QK4_Contender/train_gpt.py`
- Changes vs vault: QK_GAIN_INIT=4.0 only
- **POLLUTION: Runner defaulted WARMDOWN_ITERS=2000 instead of baseline 3500**

| metric | value |
|---|---|
| post_ema_bpb | 1.1374 |
| model int6+zstd | 16,465,239 bytes |
| total int6+zstd | 16,583,757 bytes |
| roundtrip_bpb | 1.14539983 |
| sliding_bpb | 1.11386773 |

### Verdict: INVALID AS QK4 A/B
- Worse than safepoint on every metric
- But WARMDOWN was wrong (2000 vs 3500)
- Cannot attribute failure to QK4 because two variables changed
- **The clean QK4 test (with correct warmdown=3500) has NOT been run**

---

## Critical Open Questions

1. **Does pure QK4 with WARMDOWN_ITERS=3500 work at full scale?**
   - The only full-run QK4 test had warmdown pollution
   - We genuinely do not know if QK4 helps or hurts at full scale
   - This is the single highest-priority test to run next

2. **Why do 2k proxy size numbers not predict full-run sizes?**
   - Proxy ctrl: 16.1MB, Full-run ctrl-class: ~15.5MB (safepoint)
   - Proxy qk4: 15.1MB (best), Full-run qk4+warmdown2000: 16.4MB (worst)
   - The proxy is too short / different regime to predict compression

3. **Warmdown shape experiments still unrun**
   - swirl, cascade, jitter from Rat Rod hypotheses
   - These are zero-cost training-schedule changes
   - Setup exists in `neural/experiments/QK4_Warmdown/` but NOT validated

---

## Dead Ends Confirmed Today

- SLOT in-process integration: causes size blowup at full scale
- Short size harnesses: not predictive of full-run compression
- bigram2816 alone: bad on size
- WARMDOWN_ITERS=2000 in full run: worse than baseline (likely too aggressive)

## Dead Ends Confirmed Previously (from junkyard synthesis)

- Trigram: wash
- Complementary training: worse
- Value residual: worse
- Synapse/HS-MTP bridge: dead
- Siphon: dead
- Crawler recurrence: closed as primary SOTA path
- Skip-gram: negligible

---

## What Is Still Worth Running

Ranked by confidence:

1. **Pure QK4 with WARMDOWN_ITERS=3500** — the clean test we never got
2. **Warmdown shape experiments** (swirl, cascade, jitter) — zero-cost schedule changes
3. **QK4 + warmdown shape combos** — if both show independent signal
4. **GPTQ timing budget investigation** — can we afford it within 600s?

## Files Created Today

- `experiments/rascal_hunt_2k/` — 2k proxy hunt lane
- `experiments/rascal_hunt_2k/results/2026-04-02_4gpu_build_seed444.md` — proxy sweep results
- `experiments/slot_fix_spark/` — SLOT size hunt lane (Spark)
- `experiments/COMPREHENSIVE_RESEARCH_SYNTHESIS_2026-04-02.md` — full junkyard synthesis
- `neural/experiments/Lucky II/` — FAILED QK4+SLOT contender
- `neural/experiments/QK4_Contender/` — QK4-only contender (runner NOW fixed to warmdown=3500)
- `neural/experiments/QK4_Warmdown/` — warmdown shape test kit

## Chain-of-Custody Notes

- Vault file verified intact: SHA256 `4fa8f925be9828425adb0d25934917eaffd9cf171c0410d6cf2dc58c28104acd`
- `1.110_15.5mb_baseline.py` is byte-identical to vault (confirmed by diff and git hash-object)
- `QK4_Contender/train_gpt.py` differs from vault by exactly ONE line: `QK_GAIN_INIT` default 1.5 -> 4.0
- `QK4_Contender/run.sh` was initially polluted with WARMDOWN_ITERS=2000, now fixed to 3500
- Lucky II file was vault + QK4 + SLOT eval helper (not based on Spark file)
