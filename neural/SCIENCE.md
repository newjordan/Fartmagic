# Neural Science Board

Track: Rascal lineage · Goal: beat leaderboard #1 · Score: sliding-window BPB
Champion: **1.10986874 BPB** (seed 444) · **15.44MB** · `neural/2026-03-30_Rascal_II/`

Legend: → PROMOTED · ✓ PASS · ✗ FAIL · ⏳ PENDING · — n/a

---

## Thread: Rascal Architecture — XSA + Parallel Muon + Bigram

The core lineage. Rascal II is the current leaderboard #1 (as of 2026-03-30).

| Date | Leg | Change vs Parent | Gate | Full Run BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------------------|------|---------|-------------|
| 2026-03-30 | **Rascal_II** (CHAMPION) | 11L XSA-all + Parallel Muon + Coprime loader + Bigram2048 + RoPE16 + Late QAT (step ~6070, scale=0.15) + SWA (~5900) | confirmed | **1.10986874** | **15.44MB** | → PROMOTED | 3-seed mean 1.1099. 26.99M params. SKIP_GPTQ=1 (naive int6 + zstd). 6593 steps @ ~91ms/step. |

Seed detail:
| Seed | BPB | Size |
|------|-----|------|
| 42   | 1.11018163 | 15,540,001 B |
| 300  | 1.10979099 | 15,542,719 B |
| 444  | 1.10986874 | 15,554,053 B |
| mean | **1.1099**  | 15,554,053 B (max) |

Locked architecture params (do NOT change without explicit hypothesis):
- BIGRAM_DIM=128, XSA_LAST_N=11, ROPE_DIMS=16
- COMPILE_FULLGRAPH=1
- SKIP_GPTQ=1 (baseline lane)

---

## Thread: Quantization

Hypothesis space: can we improve the naive int6 quantization path (SKIP_GPTQ=1) while staying ≤ 16MB?

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| — | (none yet) | — | — | — | — | NOT STARTED | Rascal_II is the baseline. Gate target: beat 1.10986874 @ seed 444. |

Open questions:
- Does a lower Late QAT scale (< 0.15) reduce quant gap without hurting raw BPB?
- Does applying GPTQ to more layers change the quant gap vs artifact size trade-off?
- Is there a warmdown timing that lets the model better absorb QAT?

---

## Thread: Architecture Capacity

Hypothesis space: Rascal_II has ~15.5MB of headroom — 0.5MB below the 16MB cap. Can we use it?

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| — | (none yet) | — | — | — | — | NOT STARTED | 0.5MB headroom available. Any capacity increase risks the size gate. |

Open questions:
- Extra bigram vocab (2048 → 4096)? Size risk.
- Deeper RoPE (16 → 32)? Speed risk.
- An extra XSA layer? Size + speed both at risk.

---

## Thread: Training Schedule

Hypothesis space: SWA / warmdown / QAT timing changes.

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| — | (none yet) | — | — | — | — | NOT STARTED | Current: SWA ~step 5900, QAT ~step 6070, 6593 steps total. |

---

## Planned Hypotheses

| Priority | Hypothesis | Thread | Rationale |
|----------|-----------|--------|-----------|
| TBD | First new leg off Rascal_II | TBD | What does the team want to test next? |

Gate target for all new legs: beat **1.10986874** BPB on seed 444 → confirm on seed 300.

---

## All-Time Reference

| Leg | BPB (seed 444) | Size | Mean BPB | Status |
|-----|----------------|------|----------|--------|
| (pre-Rascal experiments in junkyard) | — | — | — | — |
| **Rascal_II** | **1.10986874** | **15.44MB** | **1.1099** | **CHAMPION** |
