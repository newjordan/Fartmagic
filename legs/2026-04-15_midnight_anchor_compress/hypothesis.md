# Hypothesis — 2026-04-15_midnight_anchor_compress

Parent: legs/2026-04-14_midnight_iii_v_2_2_ckpt_eval/train_gpt.py

## One Variable
- Name: `clean offline post-build compression/eval`
- Change: load a finished Midnight Anchor checkpoint via `INIT_MODEL_PATH`, keep `ITERATIONS=0`, run offline post-EMA GPTQ only, and score only the deserialized deployed artifact.

## Why
- This isolates compression/eval from model build quality.
- It removes online GPTQ training interference and the legacy ngram tail from final scoring.

## Gate Pass Criteria
- Fail fast if `INIT_MODEL_PATH` is missing or `ITERATIONS` / `WARMUP_STEPS` are nonzero.
- Emit `DIAGNOSTIC post_ema`, `final_quant_roundtrip_exact`, and `final_sliding_window_exact` from the reloaded quantized artifact.
