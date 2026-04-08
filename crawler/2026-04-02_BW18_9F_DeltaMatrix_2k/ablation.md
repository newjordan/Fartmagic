# BW18_9F_DeltaMatrix_2k — Ablation Log

Status: ready

Primary one-command launch:

```bash
bash scripts/run_bw18_9f_matrix.sh
```

Direct launch (same run):

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BW18_9F_DeltaMatrix_2k/run_ablation_sequence.sh
```

Optional controls:

```bash
# skip QUICK, force specific FULL replays
SEED=444 NPROC_PER_NODE=4 RUN_QUICK=0 RUN_FULL=1 FULL_SOURCE_ARMS_CSV=D00,D02,D06,D07,D24 \
bash scripts/run_bw18_9f_matrix.sh

# include loop-aware GPTQ in calibration stage
SEED=444 NPROC_PER_NODE=4 RUN_LOOP_AWARE_GPTQ=1 bash scripts/run_bw18_9f_matrix.sh
```

Outputs:
- Summary: `crawler/2026-04-02_BW18_9F_DeltaMatrix_2k/results/summary_s<seed>_<timestamp>.tsv`
- Logs: `crawler/2026-04-02_BW18_9F_DeltaMatrix_2k/results/*.log`
- FULL checkpoints: `crawler/2026-04-02_BW18_9F_DeltaMatrix_2k/results/BW18F-*_s<seed>_<timestamp>.final_model.pt`

Table columns in summary:
- `stage`: `QUICK` | `FULL` | `CALIBRATION`
- `lane`: `WINDOW` or `POST_WINDOW`
- `must_retrain`: `1` for WINDOW, `0` for POST_WINDOW
- `source_ckpt`: frozen checkpoint used by post-window calibration arms
- `exit_code`: non-zero means arm failed and should not be promoted

Promotion guidance:
1. Pick best FULL WINDOW arm by lowest `int6_sw_bpb` with acceptable `step_ms` and `bytes`.
2. On that checkpoint, pick best CALIBRATION arm only if it beats `BW18Q-00` clearly.
3. Promote winner to 8x full run package after this matrix is complete.
