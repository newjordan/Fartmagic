# Medusa: Canonical FLA DeltaNet + Naive Int6

**val_bpb: PENDING** (3-seed mean) | **~9MB** | 8xH100 SXM | Successor to PR #990 (ClownCar, 1.1813)

> **Catalyst:** PR #875 (@shalyhinpavel, Pure Neural GDN, 1.0226 BPB) proved that Gated DeltaNet
> is the dominant architecture for this competition — an explosive 0.097 BPB leap over prior SOTA
> with zero TTT or caching. Medusa's DeltaNet integration is directly symbiotic with that discovery:
> the same `chunk_delta_rule` kernel that powers GDN's gated state updates is what unlocks
> Medusa's performance inside the crawler topology. Different architectures, same foundational mechanism.

## Results

| Seed | BPB (sliding window) | Size (int6+zstd) | Post-EMA BPB | Steps |
|------|---------------------:|-----------------:|-------------:|------:|
| 1337 | PENDING | PENDING | PENDING | PENDING |
| 42   | PENDING | PENDING | PENDING | PENDING |
| 123  | PENDING | PENDING | PENDING | PENDING |
| **Mean** | **PENDING** | **PENDING** | | |
| **Std dev** | **PENDING** | | | |

## What Changed vs PR #990 (ClownCar)

| Change | Reason |
|--------|--------|
| `DELTA_NET_HEADS=4` | Canonical FLA DeltaNet enabled (vs 0 in ClownCar) |
| `SKIP_GPTQ=1` | Naive int6 instead of GPTQ — ClownCar_II showed GPTQ degrading 0.7278→0.9340 (0.2062 BPB gap) due to DeltaNet state matrix outliers |
| `new_state.to(dtype)` in train_gpt.py | Fixes chunk_delta_rule returning Float32 state in BF16 training, which caused torch.compile recompiles on every rank at eval time |

## Architecture

- **Topology**: 4 flat layers + 1 crawler layer × 4 loops
- **INST_DIM**: 32 (flow instructions)
- **DeltaNet**: 4 heads, canonical `chunk_delta_rule` from `fla.ops.delta_rule`
- **Quantization**: Naive int6 (SKIP_GPTQ=1) + CRAWLER_QUANT_INT8=1
- **Dims**: XSA_LAST_N=11, BIGRAM_VOCAB_SIZE=2048, ROPE_DIMS=16
- **Schedule**: WARMDOWN_ITERS=2000, SWA_EVERY=50, LATE_QAT_THRESHOLD=0
- **N-gram eval**: DISABLED (sliding window only)

## Legality

1. No n-gram eval — sliding window only
2. No val data used during training
3. Score-first protocol not applicable (no n-gram cache)
4. int6 quantization runs inside training wallclock

## Reproduce

```bash
SEED=1337 bash experiments/Medusa/run.sh
SEED=42 bash experiments/Medusa/run.sh
SEED=123 bash experiments/Medusa/run.sh
```

8xH100 SXM, 600s training per seed.

## Credits

- **Gated DeltaNet (GDN) — primary catalyst**: @shalyhinpavel (PR #875) — proved GDN is the architecture for this competition, achieving 1.0226 BPB pure neural. Medusa's DeltaNet integration is directly symbiotic: same `chunk_delta_rule` mechanism, applied inside the crawler topology.
- **Canonical DeltaNet kernel**: `fla.ops.delta_rule` (flash-linear-attention)
- **Crawler architecture + flow instructions**: @newjordan (PR #990)
- **FX_Wing_Delta base**: @newjordan
