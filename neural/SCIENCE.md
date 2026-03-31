# Neural Science Board

Track: Rascal lineage · Goal: beat leaderboard #1 · Score: sliding-window BPB
Champion: **1.10986874 BPB** (seed 444) · **15.44MB** · `neural/2026-03-30_Rascal_II/`

Legend: → PROMOTED · ✓ PASS · ✗ FAIL · ⏳ PENDING · — n/a

---

## Competitive Landscape (updated 2026-03-31)

| Status | PR | Score (seed 444) | Author | Key Techniques | Notes |
|--------|-----|-----------------|--------|---------------|-------|
| MERGED #1 | #1019 | 1.1147 | abaybektursun | AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112 | Official leaderboard top |
| **OUR OPEN PR** | **#1120** | **1.10987** | **Frosty40** | **Rascal II — XSA-all + Muon + Bigram2048 + SKIP_GPTQ** | **Pending merge. Beats all below.** |
| Open — beats us | #1089 | **1.1091** | mikeapedia | Turbo-Muon + EngramLite + ParamBanking + ASQU | ⚠️ 0.00077 BPB ahead of us |
| Open — we beat | #1179 | 1.1105 | dexhunter | Split-LR + BigramHash 2816×160 + Brotli | Clean |
| Open — we beat | #1135 | 1.1116 | barneywohl | Fused Triton MLP + Full GPTQ + Coprime + BH2816 | Clean |
| Open — we beat | #1169 | 1.1126 | Bortlesboat | Turbo-Muon + EngramLite + ParamBanking + GPTQ Reserve | Clean |
| Open — we beat | #1060 | 1.1122 | dexhunter | Coprime-stride loader + Full Hessian GPTQ + XSA-all | Clean |
| ILLEGAL | #1176 | 1.0914 | — | SLOT + QK-Gain | SLOT ruled illegal (causality violation) |
| ILLEGAL | #1172 | 1.1015 | — | SLOT | SLOT ruled illegal |
| CONTESTED | #1185 | 0.9641 | — | N-gram backoff cache | Under dispute — likely invalid probability distributions |

**Summary**: We hold the best legal score in the open PR queue. PR #1089 at 1.1091 is the only clean
competitor ahead of us, by 0.00077 BPB — within 1-sigma seed variance.

---

## What Rascal II Has (already in stack — no need to add)

| Feature | Our Config | Notes |
|---------|-----------|-------|
| LeakyReLU(0.5)² | ✅ Yes, custom Triton kernel | Lines 151-206 in vault file |
| LN_SCALE=1/√(layer+1) | ✅ Default=1 | Matches PR #1019 |
| XSA on all 11 layers | ✅ XSA_LAST_N=11 | Matches leaders |
| Full Hessian GPTQ code | ✅ Exists (lines 552-643) | **DISABLED** — SKIP_GPTQ=1 |
| Coprime loader | ✅ Exists | COPRIME_MAX_LOADED_SHARDS=**1** (CRITICAL — do NOT change) |
| Multiple LR groups | ✅ HEAD_LR, MATRIX_LR, EMBED_LR | Leaders have similar |
| WARMDOWN_ITERS | ✅ 3500 | Leaders use 4000 — gap exists |

---

## What We Are Missing vs Competition Leaders

| Feature | Our State | Leader State | Est. BPB Delta | Risk |
|---------|-----------|-------------|---------------|------|
| Full Hessian GPTQ | SKIP_GPTQ=1 | Enabled | **−0.003 to −0.009** | Medium — costs ~328 training steps |
| AR self-gen GPTQ calibration | Training data | Self-generated seqs | ~−0.001 to −0.003 | Low once GPTQ is on |
| BigramHash vocab | 2048 | 3072 | ~−0.001 to −0.002 | Low — size est. +~31KB |
| Warmdown iters | 3500 | 4000 | ~−0.0005 | Very low |
| Brotli compression | zstd-22 | Brotli-11 | Frees artifact budget | Medium — new dependency |
| Code minification | 118,521 bytes | ~28-30KB | Frees ~88KB for weights | Medium — must still run |

Budget: 15,554,053 / 16,000,000 = **445,947 bytes headroom**.
Code: 118,521 bytes. Model: 15,435,532 bytes.

---

## Thread: Rascal Architecture — XSA + Parallel Muon + Bigram

Core lineage. Rascal II is the current best legal open submission.

| Date | Leg | Change vs Parent | Gate | Full Run BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------------------|------|---------|-------------|
| 2026-03-30 | **Rascal_II** (CHAMPION) | 11L XSA-all + Parallel Muon + Coprime (SHARDS=1) + Bigram2048×128 + RoPE16 + Late QAT + SWA | confirmed | **1.10986874** | **15.44MB** | → PROMOTED | 3-seed mean 1.1099. 26.99M params. SKIP_GPTQ=1 naive int6 + zstd-22. 6593 steps @ ~91ms. |

Seed detail:
| Seed | BPB | Size |
|------|-----|------|
| 42   | 1.11018163 | 15,540,001 B |
| 300  | 1.10979099 | 15,542,719 B |
| 444  | 1.10986874 | 15,554,053 B |
| mean | **1.1099**  | 15,554,053 B (max) |

DO NOT CHANGE without explicit hypothesis:
- BIGRAM_DIM=128, XSA_LAST_N=11, ROPE_DIMS=16
- COPRIME_MAX_LOADED_SHARDS=**1** (changing to 4 caused LC4-class failure previously)
- COMPILE_FULLGRAPH=1

---

## Thread: Quantization — GPTQ

Biggest single gap vs competition. GPTQ code is already in the vault script (lines 552–643).
We run SKIP_GPTQ=1 because original Rascal I was too large with GPTQ. Rascal II is 15.44MB —
with GPTQ enabled, quantization quality improves, potentially offsetting the ~328 lost training
steps from the 30s reserve window.

Current calibration (when GPTQ enabled): 256 samples from training data, 2048 token context.
PR #1019 uses AR self-generated data (64 seq × 2048 tok, temp=0.8) — better for deployment
distribution; does NOT touch val data (legal).

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| — | Rascal_III_GPTQ | SKIP_GPTQ=0 (enable GPTQ, training-data calib) | — | — | — | NOT STARTED | Costs ~30s → ~328 fewer steps. GPTQ_RESERVE_MS=30000. Single variable. |
| — | Rascal_III_ARcal | AR self-gen calibration (replace training-data) | — | — | — | NOT STARTED | Requires ~20 lines new code. Do AFTER GPTQ gate passes. |

---

## Thread: Architecture Capacity — Bigram Hash

Competition moved from BigramHash 2048 → 3072 (PR #1019 uses 3072×112, we use 2048×128).
More buckets = better coverage of the 2-gram space = less hash collision.
Size impact of 3072 (keep DIM=128): +1024 buckets × 128 dim = +131K params × 0.75 bytes/param × ~0.5 zstd ≈ +~50KB. Well inside 445KB headroom.

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| — | Rascal_III_Bigram3072 | BIGRAM_VOCAB_SIZE=3072 (keep DIM=128) | — | — | — | NOT STARTED | Single variable. Est. +50KB size increase. |

---

## Thread: Training Schedule

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| — | Rascal_III_Warmdown4k | WARMDOWN_ITERS=4000 (was 3500) | — | — | — | NOT STARTED | Single variable. More steps in the cool-down. est. ~0.0005 BPB. |

---

## Thread: Artifact Compression

Low-risk infrastructure wins. Brotli-11 vs zstd-22; code minification.
Code minification potential: 118KB → ~28KB = ~90KB freed for model weights.

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| — | Rascal_Brotli | Brotli-11 instead of zstd-22 | — | — | — | NOT STARTED | New python dep (brotli). Run AFTER architecture wins are locked in. |
| — | Rascal_Minified | Minify train_gpt.py (~90KB freed) | — | — | — | NOT STARTED | Infrastructure change. Minified code must be tested locally first. |

---

## Recommended Hypothesis Order

Priority based on expected BPB gain per complexity of change:

| Priority | Leg Name | Change | Expected Gain | Est. Cost |
|----------|---------|--------|--------------|-----------|
| **1** | **Rascal_III_GPTQ** | SKIP_GPTQ=0, training-data calib | −0.003 to −0.009 BPB | 1 env var |
| **2** | **Rascal_III_ARcal** | AR self-gen GPTQ calib (after GPTQ passes) | −0.001 to −0.003 more | ~20 lines code |
| **3** | **Rascal_III_Bigram3072** | BIGRAM_VOCAB_SIZE=3072 | −0.001 to −0.002 | 1 env var |
| 4 | Rascal_III_Warmdown4k | WARMDOWN_ITERS=4000 | ~−0.0005 | 1 env var |
| 5 | Rascal_Brotli | zstd → Brotli-11 | Frees budget | New dep |
| 6 | Rascal_Minified | Code minification | Frees ~90KB | Infra work |

Gate target for all new legs: beat **1.10986874** BPB on seed 444 → confirm on seed 300.

---

## All-Time Reference

| Leg | BPB (seed 444) | Size | Mean BPB | Status |
|-----|----------------|------|----------|--------|
| (pre-Rascal history — junkyard) | — | — | — | — |
| **Rascal_II** | **1.10986874** | **15.44MB** | **1.1099** | **CHAMPION (open PR #1120)** |
