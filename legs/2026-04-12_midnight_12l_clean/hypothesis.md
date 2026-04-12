# Hypothesis — 2026-04-12_midnight_12l_clean

Parent: vault/train_gpt_midnight_12l_sota_REAL.py

## Campaign Question
- Can Midnight 12L absorb `SP8192` without paying away the clean quantization behavior that made it the official leader?

## Working Facts
- Midnight 12L is the current official leader, but it is already at `15,631,603` bytes, leaving only about `368 KB` of headroom.
- In this codebase, the straight `SP8192` swap adds about `4.59M` vocab-facing params because both `tok_emb` and `ve_shared` scale with vocab.
- That means Midnight 12L cannot fit `SP8192` unchanged; a viable 12L-family lane must first remove auxiliary vocab costs and likely buy back physical depth.
- Midnight 12L is still worth testing because its quant gap is much cleaner than the III family.

## Ordered Hypotheses
| ID | Relative Parent | One Variable | Why This Is Worth Testing | Cheapest Proof |
|----|-----------------|--------------|---------------------------|----------------|
| `12L-A` | `midnight_12l_clean` | `SP8192` regime (`VOCAB_SIZE`, `DATA_PATH`, `TOKENIZER_PATH`) | Raw direction check; likely an immediate size failure, but we need to see it cleanly | Static budget or micro gate |
| `12L-B` | `12L-A` | `VE_ENABLED=0` | Largest vocab-adjacent payback in the 12L family | Quant/byte gate |
| `12L-C` | `12L-B` | `BIGRAM_VOCAB_SIZE=0` | Makes the vocab spend mostly about the actual token table | Quant/byte gate |
| `12L-D` | `12L-C` | `NUM_LAYERS=11` | The only realistic 12L-family fit lever that preserves the clean quantization lane | Quant/byte gate |
| `12L-E` | `12L-D` | `QUANT_OTHER_BITS=6` | Emergency byte trim only if `12L-D` is close and the quality looks good | Quant/byte gate |

## Stop Conditions
- Do not full-run a 12L SP8192 branch unless the quant gate is comfortably legal on bytes.
- If `12L-D` is still over `16,000,000` bytes, stop the 12L family and move on.
- Stop any 12L-family branch whose quant gap exceeds about `0.03`; 12L should stay quant-clean.
- Prefer the III family for larger-vocab work if 12L needs too many byte-saving amputations to fit.
