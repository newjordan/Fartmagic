# Ablation Log — Midnight III SP8192 Campaign

## Structure Rule

- Each arm below must become its own leg folder with its own `hypothesis.md`, `gate.sh`, `run.sh`, `ablation.md`, and `RESULTS.md`.
- Do not launch these tests by typing ad hoc env overrides against `midnight_iii_clean`.
- The tested variable must be baked into the child leg scripts so the run is self-describing in logs and review files.

## Ordered Ladder

| Arm | Planned Leg Name | Relative Parent | Baked Change | What We Learn | Cheapest Stage |
|-----|------------------|-----------------|--------------|---------------|----------------|
| `III-A` | `2026-04-12_midnight_iii_sp8192_a_raw` | `midnight_iii_clean` | `VOCAB_SIZE=8192`, `DATA_PATH=./data/datasets/fineweb10B_sp8192`, `TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model` | Raw vocab effect before compensating cuts | Static + pre-quant gate |
| `III-B` | `2026-04-12_midnight_iii_sp8192_b_no_ve` | `III-A` | `VE_ENABLED=0` | Whether removing `VE` pays back enough bytes without killing the gain | Quant/byte gate |
| `III-C` | `2026-04-12_midnight_iii_sp8192_c_no_bigram` | `III-B` | `BIGRAM_VOCAB_SIZE=0` | Whether pure token vocab beats token+aux vocab tables | Quant/byte gate |
| `III-D` | `2026-04-12_midnight_iii_sp8192_d_no_loops` | `III-C` | `NUM_LOOPS=0` | Whether the quant problem is recurrence-driven | Quant/byte gate |
| `III-E` | `2026-04-12_midnight_iii_sp8192_e_11l` | `III-C` | `NUM_LAYERS=11` | Whether a leaner physical stack can preserve the III-family upside | Quant/byte gate |

## Economical Data Capture

1. Static budget screen, no GPU
   - Reject obviously impossible branches before touching a pod.
   - Known size levers in this family:
     - Straight `SP8192` adds about `4.59M` vocab-facing params.
     - `VE_ENABLED=0` buys back about `1.08M` params at `SP8192`.
     - `BIGRAM_VOCAB_SIZE=0` buys back about `0.33M` params.

2. Pre-quant gate, 1xGPU, seed `444`
   - Use each child leg's own `gate.sh` for the cheap direction check.
   - Record: `model_params`, step time, and the last gate-time validation metric.
   - Compare only against the same-family control; do not compare 2000-step gates to full 600s runs.

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
   - Only after the quant gate is legal on bytes and reasonably clean on quant gap.
   - First expensive metric to care about: `final_sliding_window_exact`.

5. TTT
   - Only after a branch is already legal and promising on sliding-window eval.
   - TTT is not a discovery tool here; it is a late-stage amplifier.

## Suggested Stop Conditions
- Over `16,000,000` bytes at the quant gate: stop.
- Quant gap above about `0.05`: stop that III branch.
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
