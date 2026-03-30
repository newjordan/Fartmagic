# Rascal_Turbo

Rascal II copy with **TurboMuon-only** injected.

What changed vs Rascal II baseline:

- Newton-Schulz path switched to AOL + Polar coefficients (`NS4` default).
- Added post-NS normalization hook (`MUON_POST_NORM`, default `row_col`).
- No EngramLite changes in this folder.

## Full Race Run (default 600s)

```bash
bash experiments/Rascal_Turbo/run.sh
```

Common overrides:

```bash
SEED=444 NPROC_PER_NODE=8 TORCHRUN_BIN=torchrun bash experiments/Rascal_Turbo/run.sh
```

## Single-H100 2000-step Signal

```bash
bash experiments/Rascal_Turbo/run_h100_2000.sh
```

Common overrides:

```bash
SEED=444 NPROC_PER_NODE=1 ITERATIONS=2000 TORCHRUN_BIN=torchrun \
bash experiments/Rascal_Turbo/run_h100_2000.sh
```
