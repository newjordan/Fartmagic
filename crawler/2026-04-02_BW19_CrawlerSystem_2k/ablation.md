# BW19_CrawlerSystem_2k — Ablation Log

Status: ready

Primary one-command launch:

```bash
bash scripts/run_bw19_crawler_system.sh
```

Direct launch (same run):

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BW19_CrawlerSystem_2k/run_ablation_sequence.sh
```

Optional controls:

```bash
# QUICK only (crawler interaction scan)
SEED=444 NPROC_PER_NODE=4 RUN_QUICK=1 RUN_FULL=0 RUN_CALIB=0 \
bash scripts/run_bw19_crawler_system.sh

# FULL replay on chosen crawler arms
SEED=444 NPROC_PER_NODE=4 RUN_QUICK=0 RUN_FULL=1 RUN_CALIB=1 \
FULL_SOURCE_ARMS_CSV=C00,C02,C06,C09,C10,C14 \
bash scripts/run_bw19_crawler_system.sh

# include loop-aware GPTQ in calibration
SEED=444 NPROC_PER_NODE=4 RUN_LOOP_AWARE_GPTQ=1 \
bash scripts/run_bw19_crawler_system.sh
```

Outputs:
- Summary: `crawler/2026-04-02_BW19_CrawlerSystem_2k/results/summary_s<seed>_<timestamp>.tsv`
- Logs: `crawler/2026-04-02_BW19_CrawlerSystem_2k/results/*.log`
- FULL checkpoints: `crawler/2026-04-02_BW19_CrawlerSystem_2k/results/BW19F-*_s<seed>_<timestamp>.final_model.pt`

Summary columns (crawler-aware):
- `int6_sw_bpb`
- `bytes`
- `bytes_mb`
- `size_per_bpb_mb`
- `bpb_x_mb`
- `delta_vs_control`
- `delta_bytes_vs_control`

Promotion guidance:
1. Keep only non-dominated QUICK arms (quality, size, speed).
2. Replay those in FULL stage.
3. Apply post-window quant policy sweep on best FULL checkpoint.
4. Promote winner to 8x contender packaging.
