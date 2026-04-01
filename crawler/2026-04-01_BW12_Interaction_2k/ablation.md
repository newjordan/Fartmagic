# BW12_Interaction_2k — Ablation Results

Status: pending run

Run command:

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-01_BW12_Interaction_2k/run_ablation_sequence.sh
```

Summary file is emitted to:

- `crawler/2026-04-01_BW12_Interaction_2k/results/summary_s<seed>_<timestamp>.tsv`

## Lane A — WINDOW (must retrain)

| Arm | Description | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_control | Verdict |
|-----|-------------|---------|-------------|---------|-------|------------------|---------|
| BW12INT-00 | control (Nightcrawler 5F+TAP shared, naive int6) | pending | pending | pending | pending | — | pending |
| BW12INT-01 | tap off (isolate depth-only behavior) | pending | pending | pending | pending | pending | pending |
| BW12INT-02 | anchor dim=32 on Nightcrawler stack | pending | pending | pending | pending | pending | pending |

## Lane B — POST_WINDOW (sequential, no retrain)

These use `SKIP_TRAIN=1` and `INIT_MODEL_PATH=<control final_model.pt>`.

| Arm | Description | raw_bpb | int6_sw_bpb | step_ms | bytes | gptq_layers | delta_vs_control | Verdict |
|-----|-------------|---------|-------------|---------|-------|-------------|------------------|---------|
| BW12INT-Q0 | naive int6 on frozen control checkpoint | pending | pending | pending | pending | pending | pending | pending |
| BW12INT-Q1 | standard GPTQ on frozen control checkpoint | pending | pending | pending | pending | pending | pending | pending |
| BW12INT-Q2 | loop-aware GPTQ on frozen control checkpoint | pending | pending | pending | pending | pending | pending | pending |

## Full-Run Promotion Queue (600s, 8xH100)

Promote only if gate clears noise floor:

- WINDOW arm: `delta_vs_control <= -0.0008`
- POST_WINDOW quant arm: `delta_vs_control <= -0.0008`, then rerun same quant policy with full training (`SKIP_TRAIN=0`)

| Candidate | Triggered By | Full-run status |
|-----------|--------------|-----------------|
| pending | pending | pending |

## Notes

- This leg intentionally splits tests into:
  - retrain-required interaction tests,
  - post-window sequential quant tests for efficiency.
- All arms are 2000-step signal gates, not promotion by themselves.
