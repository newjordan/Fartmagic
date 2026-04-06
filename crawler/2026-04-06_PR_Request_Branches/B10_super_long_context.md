# B10: Super Long Context

## Base

- Base lanes:
  - `crawler/2026-04-06_Ouroboros_III/`
  - `crawler/2026-04-06_Helix_ab_3/`
  - `crawler/2026-04-06_TTT_ablation/`

## Status (2026-04-06)

- Long-context support exists via `TRAIN_SEQ_LEN` / `EVAL_SEQ_LEN` and sliding-window eval.
- Added dedicated wrappers so long-context runs are one-command reproducible.

## TV0 (Ouroboros + SSM + LongCtx)

```bash
SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-06_Ouroboros_III/run_ssm_longctx.sh
```

## TV1 (Helix Gate + SSM + LongCtx)

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-06_Helix_ab_3/gate_ssm_longctx.sh
```

## TV2 (TTT Gate + LongCtx)

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-06_TTT_ablation/gate_longctx.sh
```

## Default Long-Context Envelope

- `TRAIN_SEQ_LEN=4096`
- `EVAL_SEQ_LEN=4096`
- `EVAL_STRIDE=128`
- `TRAIN_BATCH_TOKENS=393216`
- `VAL_BATCH_SIZE=262144`

## Success Criteria

- Runs complete at long context with valid `final_int6_sliding_window_exact` metrics.
- Compare long-context quality/latency against 2048-token controls.
