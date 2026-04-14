# Hypothesis — 2026-04-12_midnight_iii_v_submission

Parent: vault/train_gpt_midnight_iii_base.py

## Target
- Build a submission-capable Midnight III variant (`Midnight III.V`) that carries `SP8192` while preserving effective depth via recurrence.

## User-Approved Profile (Multi-Variable)
- Note: this leg intentionally uses a wider change set for a submission profile.
- `VOCAB_SIZE=8192`
- `DATA_PATH=./data/datasets/fineweb10B_sp8192`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model`
- `NUM_LAYERS=11` (buy byte room)
- `NUM_LOOPS=2`, `LOOP_START=3`, `LOOP_END=5` (retain effective depth)
- `VE_ENABLED=0`
- `BIGRAM_VOCAB_SIZE=0`
- Full run uses GPTQ (`SKIP_GPTQ=0`) for submission-capable artifact generation.

## Why
- The SP8192 swap is expensive on vocab-facing parameters.
- Reducing physical layers while keeping looped reuse is the intended trade: byte room from shallower stack, effective depth from recurrence.
- Disabling `VE` and `BIGRAM` removes auxiliary vocab taxes that are not core token embedding capacity.

## Pass Criteria
- Quant/full run artifact is legal under submission byte cap.
- Quant gap remains controlled for III-family stop criteria.
- Sliding-window metric is competitive with current III baseline.
