# Hypothesis — 2026-04-12_midnight_iii_clean

Parent: vault/train_gpt_midnight_iii_base.py

## Campaign Question
- Can Midnight III absorb `SP8192` by stripping vocab-adjacent baggage before touching the core recurrence stack?

## Working Facts
- Midnight III is the best larger-vocab starting point in this repo because it has materially more artifact headroom than Midnight 12L.
- In this codebase, `1024 -> 8192` grows both `tok_emb` and `ve_shared`, so the straight vocab swap adds about `4.59M` vocab-facing params.
- PR #1493 fits `SP8192` mainly by staying lean: `11` physical layers plus recurrence, not by making a `15L`-class model magically cheap.
- Our first `15L + SP8192` try failed on both bytes and BPB, so the next attempts must be selective and size-aware.

## Ordered Hypotheses
| ID | Relative Parent | One Variable | Why This Is Worth Testing | Cheapest Proof |
|----|-----------------|--------------|---------------------------|----------------|
| `III-A` | `midnight_iii_clean` | `SP8192` regime (`VOCAB_SIZE`, `DATA_PATH`, `TOKENIZER_PATH`) | Establish the raw lexical upside and the raw byte pain before any compensating cuts | Static budget + 1xGPU pre-quant gate |
| `III-B` | `III-A` | `VE_ENABLED=0` | `VE` is the biggest vocab-adjacent tax in the Midnight family | Quant/byte gate |
| `III-C` | `III-B` | `BIGRAM_VOCAB_SIZE=0` | Strips the remaining auxiliary vocab table and asks whether `SP8192` alone is enough | Quant/byte gate |
| `III-D` | `III-C` | `NUM_LOOPS=0` | Isolates whether recurrence is the quant blocker rather than the vocab swap | Quant/byte gate |
| `III-E` | `III-C` | `NUM_LAYERS=11` | Tests a PR #1493-sized physical depth while keeping the III-family ideas | Quant/byte gate |

## Stop Conditions
- Stop any branch that is still over `16,000,000` bytes at the quant gate.
- Stop any branch that worsens both bytes and post-EMA BPB versus its parent.
- Stop any III-family branch whose quant gap (`final_quant_roundtrip_exact - DIAGNOSTIC post_ema`) exceeds about `0.05`.
- Do not spend TTT time on a branch that is not already legal on bytes and reasonably clean on quant gap.
