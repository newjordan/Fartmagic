# PR Request Branch Incubator (2026-04-06)

This folder organizes concept-first PR branches requested in the challenge README.  
Each branch has:
- one concrete test variant (`TV0`) to run immediately or with minimal edits
- an ablation ladder (`A0..A3+`)
- a chosen base model from our strongest local runs

## Base Model Pool

| Base ID | val_bpb | Model | Path |
|---|---:|---|---|
| BASE_RASCAL | 1.1099 | Rascal | `records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py` |
| BASE_LEAKY_TTT | 1.1194 | LeakyReLU2 + Legal TTT + Parallel Muon | `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` |
| BASE_GPTQ_LITE | 1.1228 | 11L EMA + GPTQ-lite | `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` |
| BASE_BINARY | 1.1239 (non-record) | 106M Binary U-Net | `records/track_non_record_16mb/2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear/train_gpt_cuda_binary.py` |
| BASE_TERNARY | 1.1565 | 73.7M Ternary U-Net | `records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/train_gpt_cuda_ternary.py` |
| BASE_NIGHTCRAWLER | 1.1761 | Nightcrawler | `records/track_10min_16mb/2026-04-01_Nightcrawler_8xH100/train_gpt.py` |
| BASE_4H | 1.2074 (non-record) | Quasi10B 4-hour baseline | `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/train_gpt.py` |
| BASE_LONGCTX | 1.2014 | Seq4096 long-context baseline | `records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py` |

## Requested Branches

| Branch File | Requested Idea | Base |
|---|---|---|
| `B01_1bit_quant.md` | 1-bit quantization | BASE_BINARY |
| `B02_ternary_quant.md` | Ternary quantization | BASE_TERNARY |
| `B03_jepa.md` | JEPA | BASE_RASCAL |
| `B04_text_diffusion.md` | Text diffusion | BASE_RASCAL |
| `B05_hnet_tokenization.md` | H-net tokenization | BASE_RASCAL |
| `B06_universal_transformer_4h.md` | Universal Transformer (4-hour) | BASE_4H |
| `B07_megakernels.md` | Megakernels | BASE_RASCAL |
| `B08_state_space_models.md` | State-space models | BASE_NIGHTCRAWLER |
| `B09_e2e_ttt.md` | E2E TTT | BASE_LEAKY_TTT |
| `B10_super_long_context.md` | Super long context | BASE_LONGCTX + BASE_RASCAL |
| `B11_random_linear_adapters.md` | Adapters on random linear maps | BASE_LEAKY_TTT |

## Execution Order (Fastest Signal)

1. B01 and B02 (already implemented families, fastest validation of branch workflow)
2. B07 (systems-only path, low algorithmic risk)
3. B09 and B10 (existing TTT/long-context infra)
4. B03, B08, B11 (moderate code additions)
5. B04, B05, B06 (largest architectural delta)

## Standard Success Gates

- `Gate 1 (sign-of-life)`: beats naive baseline (`val_bpb < 1.2244`) at matched steps
- `Gate 2 (serious)`: within `+0.02` bpb of branch base at matched steps
- `Gate 3 (PR-worthy)`: either meaningful bpb gain or clear novel capability with reproducible run logs

