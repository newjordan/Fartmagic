# Theory Index

## Core Thesis

The strongest current lane is not a tiny global knob change. It is the whale
donor body plus disciplined compression/quantization repair:

```text
12L + dim576 + matrixbits5 + loop 3..5 + XSA_LAST_N=11
```

The donor has raw strength, but final submission quality is bounded by artifact
bytes and quantization damage.

## Working Beliefs

1. The extra width/depth contains real signal.
2. Uniform global int5-style quantization is too blunt for this body.
3. Late layers, loop layers, MLP projection/FC paths, and control tensors are
   likely more sensitive than a flat policy assumes.
4. Global clip sweeps are low leverage unless paired with layer/tensor-family
   evidence.
5. Eval-time tricks are secondary. The body must beat same-pod controls first.
6. CaseOps/tokenizer changes remain blocked until legality is explicitly
   cleared; portable PR1797 ideas can only be tested after clean donor control.

## Most Useful Next Research Questions

### 1. Clean Donor Rebuild

Question: can the April 15 donor be rebuilt from reviewed clean source and
rerun on current 4x/8x mechanics without source ambiguity?

Gate:

- logs show the same donor config;
- tokenizer hash is corrected/pinned;
- `train_shards: 80`;
- raw curve and final eval are directly comparable to current controls.

### 2. Quant Damage Localization

Question: which tensor family causes the donor/body to lose most of its raw
score after GPTQ?

High-value probes:

- global int5 same-checkpoint baseline;
- MLP projection int6 subset;
- MLP FC int6 subset;
- attention projection int6;
- QKV int6;
- layer-band policies early/mid/late;
- loop-layer-only relief.

Pass signal:

- reduced post-EMA to quantized BPB delta;
- total bytes legal or with a measured byte-recovery path.

### 3. Byte Recovery

Question: can the donor remain under `16,000,000` bytes while protecting only
the sensitive pieces?

Promising directions:

- tensor-group bit allocation;
- layer-band bit allocation;
- low-rank quant error patches;
- channel equalization before GPTQ;
- outlier row/column escape hatch;
- code/package overhead reduction before weakening model body.

### 4. Body Candidate Screen

Question: does a newer neural body like h10/h08 improve the raw curve enough to
justify rebuilding the quant stack, or should the April 15 donor remain the
primary base?

Evidence to use:

- `BASE_MODEL_DECISION_20260425.md`
- `BLESSED_WHALE_CRACK_IT_ATTACK_PLAN_20260425.md`
- `FINISHLINE_RESULTS_20260425.tsv`

Required discipline:

- compare only to same-hardware, same-data, same-evaluator controls;
- do not promote raw-only wins as final submission wins;
- require checkpoint provenance before quant ablations.

## Stop Rules

- Stop any path that cannot name its trainer checksum, tokenizer hash, world
  size, wallclock, and exact config.
- Stop any over-cap artifact without a credible byte-recovery plan.
- Stop any test whose result is not comparable to a same-pod control.
- Do not spend more time on broad PR1493/control reruns without explicit
  approval.

