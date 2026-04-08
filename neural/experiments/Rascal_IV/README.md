# Rascal IV

Rascal IV is a new campaign forked from Rascal III hardened defaults.

Purpose:
- Keep Rascal III stable and untouched while we test high-upside edges.
- Run only explicit, named arms with canonical guardrails.
- Default mode is locked to `control`; non-control arms require explicit opt-in.
- Clean runner available for stabilized/legal runs with dead-weight eval paths disabled.

Base profile (all arms inherit this unless explicitly changed):
- `LOADER_MODE=coprime`
- `COPRIME_SHARDS_PER_BATCH=1`
- `COPRIME_SHARD_HOLD_STEPS=64`
- `QK_GAIN_INIT=1.5`
- `MUON_BACKEND_STEPS=5`
- mixed-int: `attn=5, mlp=6, aux=6, embed=8, other=8`
- disabled risky toggles: `TRIGRAM=0`, `GATED_ATTENTION=0`, `VALUE_RESIDUAL=0`, `DTG_ENABLED=0`, `QAT_ENABLED=0`
- legacy eval adaptation hard-disabled: `TTT_ENABLED=0`, `TTT_EPOCHS=0`, `SCALE_TTT_ENABLED=0`, `SLOT_ENABLED=0`

## Arms (ranked queue)

1. `mtp1` (recommended first)
- Change: `MTP_NUM_HEADS=1`
- Rationale: highest near-term upside from today’s ranked edges with minimal code risk.

2. `longctx`
- Change: `TRAIN_SEQ_LEN=4096`, `EVAL_SEQ_LEN=4096`
- Rationale: tests long-context training leverage directly on the hardened base.

3. `e2e_ttt`
- Change: routes to concept script `concept_ssm_ttt/arm_b_e2e_ttt/train_gpt.py`
- Rationale: high-upside research arm; larger architecture delta than `mtp1`/`longctx`.

4. `control`
- No additional changes; hardened Rascal IV baseline.

## Usage

Control:
```bash
bash experiments/Rascal_IV/run_8x.sh
```

Clean control (recommended stable path):
```bash
bash experiments/Rascal_IV/run_8x_clean.sh
```

Clean profile details:
- forces `RASCAL_IV_ARM=control` (no ablation branches)
- `SKIP_GPTQ=1` (no GPTQ calibration pass)
- defaults `EXPORT_RESERVE_MS=0` to avoid losing training time when GPTQ and roundtrip eval are disabled
- `NGRAM_EVAL_ORDER=0` and related n-gram eval knobs off
- `QUANT_ROUNDTRIP_EVAL=0` (skip dequant roundtrip eval pass; artifact still exported)

MTP arm:
```bash
RASCAL_IV_ARM=mtp1 RASCAL_IV_ALLOW_EXPERIMENTAL=1 bash experiments/Rascal_IV/run_8x.sh
```

Long-context arm:
```bash
RASCAL_IV_ARM=longctx RASCAL_IV_ALLOW_EXPERIMENTAL=1 bash experiments/Rascal_IV/run_8x.sh
```

E2E-TTT arm:
```bash
RASCAL_IV_ARM=e2e_ttt RASCAL_IV_ALLOW_EXPERIMENTAL=1 bash experiments/Rascal_IV/run_8x.sh
```
