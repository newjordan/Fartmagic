# Smokestack — Ablation Log

Status: ready

## Gate command

```bash
SEED=444 NPROC_PER_NODE=4 bash experiments/smokestack/gate.sh
```

## Gate Results (2000 steps)

| arm | desc | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_control | verdict |
|-----|------|--------------|---------|-------------|---------|-------|------------------|---------|
| SMKSTK-00 | control: loops=3 | | | | | | +0.000000 | baseline |
| SMKSTK-01 | loops=2 | | | | | | | |
| SMKSTK-02 | loops=1 | | | | | | | |

Pass criteria: delta_vs_control <= -0.003 int6_sw_bpb

## Full Run Results (8xH100, 600s) — conditional on gate pass

| seed | arm | source | int6_sw_bpb | step_ms | bytes | vs BWX 9F (1.13867894) | verdict |
|------|-----|--------|-------------|---------|-------|------------------------|---------|
| 444 | | best gate arm | | | | | |
| 300 | | confirmation | | | | | |

Promotion gate: beat 1.13867894 on seed 444, confirm on seed 300, artifact <= 16MB.
