# BW13_TapOff_Anchor_GPTQ_2k — Ablation Results

Status: pending run

Run command:

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-01_BW13_TapOff_Anchor_GPTQ_2k/run_ablation_sequence.sh
```

Summary file is emitted to:

- `crawler/2026-04-01_BW13_TapOff_Anchor_GPTQ_2k/results/summary_s<seed>_<timestamp>.tsv`

## Lane A — WINDOW (must retrain)

| Arm | Description | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_control | Verdict |
|-----|-------------|---------|-------------|---------|-------|------------------|---------|
| BW13INT-00 | control (tap-off Nightcrawler, naive int6) | pending | pending | pending | pending | — | pending |
| BW13INT-01 | tap-off + anchor dim=32 | pending | pending | pending | pending | pending | pending |
| BW13INT-02 | tap-off + anchor dim=64 | pending | pending | pending | pending | pending | pending |

## Lane B — POST_WINDOW (sequential, no retrain)

These use `SKIP_TRAIN=1` and `INIT_MODEL_PATH=<control final_model.pt>`.

| Arm | Description | raw_bpb | int6_sw_bpb | step_ms | bytes | gptq_layers | gptq_cal_sec | delta_vs_control | Verdict |
|-----|-------------|---------|-------------|---------|-------|-------------|--------------|------------------|---------|
| BW13INT-Q0 | naive int6 on frozen control checkpoint | pending | pending | pending | pending | pending | pending | pending | pending |
| BW13INT-Q1 | standard GPTQ (128x2048) | pending | pending | pending | pending | pending | pending | pending | pending |
| BW13INT-Q1L | standard GPTQ-lite (64x1024) | pending | pending | pending | pending | pending | pending | pending | pending |

## Full-Run Promotion Queue (600s, 8xH100)

Promote only if gate clears noise floor:

- WINDOW arm: `delta_vs_control <= -0.0008`
- POST_WINDOW quant arm: `delta_vs_control <= -0.0008`, then rerun same quant policy with full training (`SKIP_TRAIN=0`)

| Candidate | Triggered By | Full-run status |
|-----------|--------------|-----------------|
| pending | pending | pending |

## Notes

- This leg is intentionally interaction-focused and efficient for 4x pods.
- `Q1L` tests whether GPTQ calibration cost can be reduced without losing score signal.
