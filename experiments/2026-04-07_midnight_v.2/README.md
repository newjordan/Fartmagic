## Midnight 12L

Midnight 12L is a 12-layer Rascal II submission that uses mixed-int quantization plus Brotli
packing to add one extra transformer layer while staying under the 16,000,000-byte artifact cap.

## Architecture summary

- Backbone: 12-layer Rascal II decoder
- Attention: GQA (`num_heads=8`, `num_kv_heads=4`)
- Context features: Bigram hash 2048, RoPE dims 16, XSA on last 11 layers
- Quantization: `attn=int5`, `mlp=int6`, `aux=int6`, `embed=int8`, `other=int8`
- Compression: mixed-int checkpoint + Brotli
- Hardware: 8xH100 SXM
- Train wallclock: 600s
- `bytes_code`: 124,698

## 3-seed results

| Seed | val_bpb_exact (sliding window) | Steps | Train time (s) | bytes_total |
|------|--------------------------------|------:|---------------:|------------:|
| 444  | 1.10567949                     | 6160  | 600            | 15631603    |
| 300  | 1.10582448                     | 6154  | 600            | 15624171    |
| 42   | 1.10641160                     | 6153  | 600            | 15619003    |
| **mean** | **1.10597186**             |       |                |             |
| **std (population)** | **0.00031653** |       |                |             |
| **max bytes_total** |                |       |                | **15631603** |

## Technique description

Compared to the prior 11-layer stack, this run spends compression headroom on depth:
the model is extended to 12 layers while preserving submission legality through mixed-int
quantization and Brotli artifact compression. Training and scoring remain standard score-first
evaluation, with no validation-set leakage.

## Reproduce

```bash
SKIP_GPTQ=1 SEED=444 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-07_Midnight_12L_8xH100/train_gpt.py
```

## v.2 ablations

- Matrix file: `experiments/2026-04-07_midnight_v.2/ablation_matrix.tsv`
- Runner: `experiments/2026-04-07_midnight_v.2/run_ablation.sh`

List lanes:

```bash
bash experiments/2026-04-07_midnight_v.2/run_ablation.sh list
```

Run one lane:

```bash
SEED=444 NPROC_PER_NODE=8 \
bash experiments/2026-04-07_midnight_v.2/run_ablation.sh warmdown_3000
```

Budget-aware profiles:

- `full` (default): full-fidelity run (~12-13 min on current stack)
- `screen`: cheap ranking mode (~4-5 min, no sliding final eval, quant-roundtrip eval skipped)
- `ultra_cheap`: minimum-cost smoke (~3 min, noisy, quant-roundtrip eval skipped)

Examples:

```bash
# Cheap screen pass
SEED=444 bash experiments/2026-04-07_midnight_v.2/run_ablation.sh warmdown_3000 screen

# Full confirm pass
SEED=444 bash experiments/2026-04-07_midnight_v.2/run_ablation.sh warmdown_3000 full
```

Suggested cost discipline:

1. Run all priority-1 lanes in `screen`.
2. Promote top 2-3 lanes to `full`.
3. Run multi-seed only for the best 1 lane.

## Batch screening (organized + economical)

Use the batch runner to execute screen lanes back-to-back with an auto summary table.

```bash
SEED=444 NPROC_PER_NODE=8 PROFILE=screen PRIORITY_MAX=1 \
bash experiments/2026-04-07_midnight_v.2/run_screen_batch.sh
```

Useful knobs:

- `BUDGET_MINUTES=45`: hard budget cap for the batch
- `PROFILE=ultra_cheap`: faster/lower-cost smoke pass
- `CONTROL_AT_END=1`: rerun control at end to detect drift
- `GATE_COMBO=1`: skip combo lane when parent lanes show no improvement
- `MAX_WALLCLOCK_SECONDS=210`: override per-lane time cap

Dry-run preview:

```bash
DRY_RUN=1 bash experiments/2026-04-07_midnight_v.2/run_screen_batch.sh
```

Adaptive embedding precision is now supported for v.2 export lanes:

- `ADAPTIVE_EMBED_PRECISION=1`
- `ADAPTIVE_EMBED_KEEP_FRAC` (fraction of highest-norm embed rows kept at high bits)
- `ADAPTIVE_EMBED_LOW_BITS` (precision used for non-kept rows)
