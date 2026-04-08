# Ablation: BW23_EcoConcept_9F

Date: 2026-04-08  
Track: crawler  
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Matrix Layout

### Window stage (retrain)

| Arm | Change | qatsurrogate | Notes |
|---|---|---|---|
| BW23W-00 | control (QAT off) | legacy | BWX-style reference |
| BW23W-01 | `QAT_ENABLED=1` | legacy | QAT baseline |
| BW23W-02 | `QAT_ENABLED=1` | softclamp | continuous surrogate |
| BW23W-03 | `QAT_ENABLED=1` | sigmoidste | smooth staircase surrogate |

### Quant stage (post-window, no retrain)

| Arm | Change | INT6_CATS | Notes |
|---|---|---|---|
| BW23Q-00 | control export policy | mlp,attn,aux | existing policy |
| BW23Q-01 | quant policy delta | mlp,attn | aux promoted to int8 |
| BW23Q-02 | quant policy delta | attn | aggressive sensitivity split |
| BW23Q-03 | quant policy delta | mlp | aggressive sensitivity split |
| BW23Q-04 | quant policy delta | all | stress full-int6 policy |

## Full Gate (2k) Run

Status: [ ] pending  [ ] pass  [ ] fail

Command:
`SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-08_BW23_EcoConcept_9F/gate.sh`

Summary TSV:
`crawler/2026-04-08_BW23_EcoConcept_9F/results/summary_gate_s<seed>_<timestamp>.tsv`

## DGX Spark Smoke Run

Status: [ ] pending  [ ] pass  [ ] fail

Command:
`SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-08_BW23_EcoConcept_9F/gate_dgx_spark_smoke.sh`

Summary TSV:
`crawler/2026-04-08_BW23_EcoConcept_9F/results/summary_smoke_s<seed>_<timestamp>.tsv`

## Notes

- Smoke run is a mirrored matrix, not a different hypothesis.
- Promote to expensive follow-up only if smoke and full gate agree directionally.
