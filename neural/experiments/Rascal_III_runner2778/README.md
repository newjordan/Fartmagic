# Rascal III Runner2778

This is the legacy Rascal III runner copied from:

- `junkyard/experiments/Rascal_III/train_gpt.py`

Canonical location:

- `neural/experiments/Rascal_III_runner2778/train_gpt.py`

Purpose:

- Preserve the high-performing pre-eval runner variant (27,780,699 params)
- Eliminate junkyard/wildcard path ambiguity
- Provide a stable launcher with explicit defaults

## Default launcher behavior

`run_8x.sh` pins:

- `LOADER_MODE=coprime`
- `COPRIME_SHARDS_PER_BATCH=1`
- `COPRIME_SHARD_HOLD_STEPS=64`
- TTT disabled by default:
  - `TTT_EPOCHS=0`
  - `TTT_LR=0.0`
  - `TTT_FREEZE_BLOCKS=0`

## Run

```bash
bash neural/experiments/Rascal_III_runner2778/run_8x.sh
```

or

```bash
bash neural/test_lab/Rascal_III_runner2778/run.sh
```

