# BW XII Quant Fix — Hypothesis

## Problem
SplitHead architecture produces excellent raw BPB (1.2012) and tiny artifacts
(11MB) but has a 0.049 quant gap (1.2503 int6_sw vs 1.2012 raw). Need to
close this gap to make BW XII competitive.

## Root Cause
Full cross-attention through shared weights creates weight distributions that
loop-aware GPTQ can't calibrate cleanly. The cross K/V path is a different
pattern than sequential self-attention.

## Fix Strategies (32 arms across 4 GPUs)

### Smart Skip (GPU 0)
Crawler-gated U-Net skip connections. The crawler already knows which layers
produce signal vs noise (it cross-attends to all of them). Gate each skip
with a confidence score from the crawler. Low-quality signal gets suppressed
before it enters the quant-sensitive decoder path.

### Weight Decay Sweep (GPU 1)
WD 0.08→0.25. Higher WD keeps shared weights closer to zero = more
quantization-friendly distributions. Find the sweet spot where BPB
doesn't degrade but quant gap shrinks.

### Stride + Cross-Head Count (GPU 2)
Fewer crawler firings (stride=2/3) = less shared-weight amplification.
Fewer cross-heads (cross=2/3) = less cross-attention pressure on weights.
Trade some BPB for quant robustness.

### Combos (GPU 3)
Stack the winners. Smart Skip + high WD + stride=2 + crawler int8.
The maximalist approach: every quant fix simultaneously.
