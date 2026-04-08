# Wildcard Theories — Last Resorts

Nuclear options. Only consider when conventional paths are exhausted.
Each carries significant risk, cost, or both. Saved here so they don't get lost.

---

## INT5 Full Model (+0.03 BPB slide, saves 3.4MB)

**Status: DEAD unless paired with something that offsets +0.03**

The slide is catastrophic — worse than SLOT 32 recovers. Only viable if:
- A breakthrough in per-layer mixed quant (int5 on insensitive layers, int6 on sensitive)
- QAT tuned specifically for int5 noise profile
- Some unknown calibration technique collapses the slide

Quant sweep data (2026-04-03 Lucky V checkpoint):
- int5 cr15: val_bpb 1.1738, slide +0.0393, size 12.1MB
- int6 cr31: val_bpb 1.1430, slide +0.0084, size 15.5MB

---

## MLP 4x at INT6 (needs ~1.2MB headroom)

**Status: BLOCKED on size. Needs code minification + compression wins.**

MLP 4x adds ~5.8M params (~4.3MB at int6). Way over budget at current compression.
Could fit if: code minification (100KB) + bank QAT compression + SP4096 embedding
trade-off balances. Speculative.

---

## Scylla / Custom Tokenizer (998 tokens)

**Status: RISKY — byte accounting under audit**

PR #1271 found Scylla BPB measurements were inflated by incorrect byte counts.
PROTEUS (1.0819) uses Scylla — score may not be real.
Only consider if byte accounting is proven correct by third party.

---

## Per-Layer Bit Allocation (mixed int5/int6/int7/int8)

**Status: UNEXPLORED — needs sweep harness**

Quant sweep showed MLP is the most sensitive (+0.005 BPB from int6→int8).
Embed is least sensitive (int8 only costs 141KB, saves 0.0015 slide).
A layer-by-layer sweep could find the optimal bit allocation that minimizes
slide while staying under 16MB. Combinatorial search — needs automation.

---

## Noisy QAT (Differentiable Uniform Noise)

**Status: PROVEN on recurrent arch (PR #363), UNTESTED on ours**

Instead of STE fake-quantize, inject calibrated uniform noise:
```
noise = (rand - 0.5) * (amax / 127.0)
w = w + noise
```
Collapsed quant gap from 0.37 to 0.002 BPB on looped transformer.
Our model isn't recurrent so the effect may be smaller, but the technique
is orthogonal to STE QAT. Could combine with bank QAT.

---

## SLOT Per-Layer Deltas

**Status: UNTESTED**

Current SLOT: single shared delta (1,1,dim) = 512 params.
What if each layer gets its own delta? 11 × 512 = 5,632 params.
More capacity for test-time adaptation. Still legal — optimizes
on context positions only. Eval time cost unclear.

---

## SLOT Learning Rate Sweep

**Status: UNTESTED**

SLOT_LR is 0.005 (never tuned). SLOT gives 0.014 BPB.
Even a 10% improvement from better LR = 0.0014 BPB free.
Cheap to test — just re-run eval with different LR on existing checkpoint.

---

## Double Warmdown (Train Twice)

**Status: INSANE**

Use first 300s to train. Checkpoint. Reset optimizer. Train 300s more
from checkpoint with fresh warmdown. Two learning curves in one budget.
Nobody has tried this. Probably terrible. But what if it isn't?
