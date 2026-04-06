# B09: E2E TTT

## Base

- Base lane: `crawler/2026-04-06_TTT_ablation/`
- Core code path: `crawler/2026-04-06_TTT_ablation/train_gpt_ttt.py`

## Status (2026-04-06)

- Implemented `TTTLayer` with inner adaptation (`autograd.grad`) in crawler and helix paths.
- Patched compile incompatibility by auto-disabling `torch.compile` when `TTT_DIM>0`.
- Patched eval compatibility so TTT runs under `no_grad` (not `inference_mode`).

## TV0 (Control Gate)

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-06_TTT_ablation/gate.sh
```

## TV1 (Long-Context TTT Gate)

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-06_TTT_ablation/gate_longctx.sh
```

Defaults in `gate_longctx.sh`:

- `TRAIN_SEQ_LEN=4096`
- `EVAL_SEQ_LEN=4096`
- `EVAL_STRIDE=128`
- `TRAIN_BATCH_TOKENS=393216`

## Knobs

- `TTT_DIM` (0 disables, 32 baseline)
- `TTT_LR`
- `TTT_STEPS`

## Success Criteria

- TTT arms complete without compile/double-backward crash.
- TTT improves `final_int6_sliding_window_exact val_bpb` vs matched control.
