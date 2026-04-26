# Key Metrics

## Active Donor

```text
id: 2026-04-15_whale_l12_dim576_matrixbits5
seed: 444
hardware in preserved run: 8x H100
tokenizer: fineweb_8192_bpe.model
train shards: 80
world size: 8
```

Architecture/config lock:

| Field | Value |
| --- | ---: |
| `NUM_LAYERS` | `12` |
| `MODEL_DIM` | `576` |
| `EMBEDDING_DIM` | `576` |
| `NUM_HEADS` | `8` |
| `NUM_KV_HEADS` | `4` |
| `MLP_MULT` | `4.0` |
| `XSA_LAST_N` | `11` |
| `MATRIX_BITS` | `5` |
| `NUM_LOOPS` | `2` |
| `LOOP_START` | `3` |
| `LOOP_END` | `5` |
| `ENABLE_LOOPING_AT` | `0.35` |
| `PARALLEL_RESIDUAL_START` | `8` |
| `TTT_ENABLED` | `1` |

Preserved donor result:

| Metric | Value |
| --- | ---: |
| raw stop | `3295/20000` |
| raw BPB | `1.0802` |
| post-EMA raw BPB | `1.07952782` |
| quantized BPB | `1.11723031` |
| quantized sliding BPB | `1.10039777` |
| quantized TTT BPB | `1.09561039` |
| serialized model bytes | `184812557` |
| quantized+brotli model bytes | `15947088` |
| total submission bytes | `15995671` |
| byte headroom under 16MB | `4329` |
| model params | `48551136` |
| peak allocated memory | `46139 MiB` |

Source evidence:

- `source_copies/donor_record/RESULTS.md`
- `source_copies/donor_record/tracked_env.sh`
- `source_copies/evidence_logs/whale_l12_dim576_matrixbits5_s444.txt`
- `source_copies/evidence_logs/full_seed444_20260416_030758.log`

## Important Comparison Anchors

Faithful PR1493 baseline from memory/current docs:

| Anchor | Metric |
| --- | ---: |
| PR1493 faithful quantized TTT BPB | `1.08036236` |
| PR1493 faithful total bytes | `16026271` |

Whale-family signals:

| Candidate | Signal | Status |
| --- | ---: | --- |
| `whale_dim576` | `1.07258990` | illegal on bytes |
| `whale_l12_dim576_matrixbits5_calib256` | `1.09552035`, `15995556` bytes | best legal branch noted in older repo memory |
| active donor `whale_l12_dim576_matrixbits5` | `1.09561039`, `15995671` bytes | current active donor lock |

Recent body-selection evidence:

| Candidate | Body | Step | Raw BPB | Params | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `h10_l11_mlp50` | 11L, MLP 5.0, xsa last 10 | `2585` | `1.0839` | `52197208` | provisional base in current docs |
| `h08_mlp50` | 12L, MLP 5.0, xsa last 11 | `2413` | `1.0840` | `56513760` | tied raw, larger/slower |
| PR1493 2x anchor | 11L, MLP 4.0 | `3636` | `1.0968` | `35944536` | same-pod control |

## Known Hazards

- The active donor record is strong but the preserved non-vault binary was not
  found in the local repo.
- New work must rebuild from clean source provenance before any paid launch.
- Do not judge 1x/2x/4x proxy BPB directly against the 8x donor record without
  a same-hardware/current-control comparison.
- Any candidate over `16,000,000` bytes is diagnostic unless it has a measured
  byte-recovery path.

