# Ablation Log — Midnight 12L SP8192 Campaign

## Structure Rule

- Each arm below must become its own leg folder with its own `hypothesis.md`, `gate.sh`, `run.sh`, `ablation.md`, and `RESULTS.md`.
- Do not launch these tests by typing ad hoc env overrides against `midnight_12l_clean`.
- The tested variable must be baked into the child leg scripts so the run is self-describing in logs and review files.

## Ordered Ladder

| Arm | Planned Leg Name | Relative Parent | Baked Change | What We Learn | Cheapest Stage |
|-----|------------------|-----------------|--------------|---------------|----------------|
| `12L-A` | `2026-04-12_midnight_12l_sp8192_a_raw` | `midnight_12l_clean` | `VOCAB_SIZE=8192`, `DATA_PATH=./data/datasets/fineweb10B_sp8192`, `TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model` | Raw vocab effect and raw size failure | Static or micro gate |
| `12L-B` | `2026-04-12_midnight_12l_sp8192_b_no_ve` | `12L-A` | `VE_ENABLED=0` | Whether the main vocab-adjacent tax was `VE` | Quant/byte gate |
| `12L-C` | `2026-04-12_midnight_12l_sp8192_c_no_bigram` | `12L-B` | `BIGRAM_VOCAB_SIZE=0` | Whether the 12L family can carry `SP8192` without auxiliary vocab tables | Quant/byte gate |
| `12L-D` | `2026-04-12_midnight_12l_sp8192_d_11l` | `12L-C` | `NUM_LAYERS=11` | Whether one less physical layer is enough to buy back legality | Quant/byte gate |
| `12L-E` | `2026-04-12_midnight_12l_sp8192_e_other_int6` | `12L-D` | `QUANT_OTHER_BITS=6` | Last-resort byte trim if the branch is close and still clean | Quant/byte gate |

## Economical Data Capture

1. Static budget screen, no GPU
   - Midnight 12L starts with only about `368 KB` of byte headroom.
   - Straight `SP8192` is therefore a budgeting exercise first, a BPB exercise second.
   - Known size levers in this family:
     - Straight `SP8192` adds about `4.59M` vocab-facing params.
     - `VE_ENABLED=0` buys back about `1.08M` params at `SP8192`.
     - `BIGRAM_VOCAB_SIZE=0` buys back about `0.33M` params.

2. Micro gate, 1xGPU, seed `444`
   - Only for `12L-A` if you want a sanity check on loss direction.
   - If the static math already says "hopelessly over", do not waste pod time here.

3. Quant/byte gate, 1xGPU, seed `444`
   - Make a dedicated quant-gate child leg if needed; do not mutate a shell command at launch time.
   - Bake these settings into that leg's scripts:
     - `ITERATIONS=2000`
     - `WARMDOWN_ITERS=500`
     - `SKIP_GPTQ=0`
     - `SKIP_FINAL_EVAL=1`
     - `POST_EMA_DIAGNOSTIC=1`
   - Record exactly these lines:
     - `DIAGNOSTIC post_ema`
     - `final_quant_roundtrip_exact`
     - `Total submission size`
     - `Code size`

4. Full run, 8xH100, seed `444`
   - Only after the quant gate is clearly under the byte cap.
   - Midnight 12L has no business doing a full SP8192 run if it is still marginal on bytes after `12L-D`.

## Suggested Stop Conditions
- Over `16,000,000` bytes after `12L-D`: stop the 12L family.
- Quant gap above about `0.03`: stop that 12L branch.
- Worse post-EMA BPB and worse bytes than parent: stop.

## Results

### Gate (1xGPU, 2000 steps)
- Command:
- Seed:
- Proxy metric:
- Verdict: PASS / FAIL
- Notes:

### Quant/Byte Gate (1xGPU, 2000 steps)
- Command:
- DIAGNOSTIC post_ema:
- final_quant_roundtrip_exact:
- Total submission size:
- Quant gap:
- Verdict: PASS / FAIL

### Full Run (8xH100, 600s, seed=444)
- Command:
- final_sliding_window_exact val_bpb:
- Delta vs leader:
- Verdict: PROMOTION_CANDIDATE / FAIL

### Confirmation (8xH100, 600s, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
