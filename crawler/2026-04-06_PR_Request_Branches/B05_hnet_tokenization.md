# B05: H-net Tokenization

## Base

- Base ID: `BASE_RASCAL`
- Path: `records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py`
- Current reference score: `val_bpb=1.1099`

## TV0 (Tokenizer Control)

```bash
cd /home/frosty40/parameter-golf-lab
SKIP_GPTQ=1 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=42 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py
```

## TV1 (H-net Tokenizer Stub; Requires Code Path)

- Add hierarchical tokenizer path with coarse/fine token streams.
- Proposed flags:
  - `HNET_TOKENIZER=1`
  - `HNET_COARSE_VOCAB=2048`
  - `HNET_FINE_VOCAB=1024`
  - `HNET_ROUTER_TOPK=2`

## Ablation Ladder

| Arm | Change | Goal |
|---|---|---|
| A0 | Existing SP1024 tokenizer | Baseline |
| A1 | Learned merge policy only | Check token-efficiency gains |
| A2 | Two-level H-net tokens (coarse + residual) | Measure bpb and length distribution |
| A3 | H-net + existing bigram hash off/on | Interaction with lexical prior |

## Code Touchpoints

- Data pipeline (`data/`): tokenizer export + shard writer
- `train_gpt.py`: multi-stream token embedding + decode head

## References (Local Notes)

- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2603.03583_byteflow-language-modeling-through-adaptive-byte-compression-without-a-tokenizer_20260401_045211.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.13940_you-can-learn-tokenization-end-to-end-with-reinforcement-learning_20260401_001947.md`
- `/home/frosty40/ml-research-analysis/ml_research_analysis_2026/2602.00594_kanade-a-simple-disentangled-tokenizer-for-spoken-language-modeling_20260210_025220.md`

