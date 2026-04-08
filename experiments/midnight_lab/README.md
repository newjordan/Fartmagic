# Midnight Lab

This folder is the organized lab harness for Midnight tests.

Protocol:
- Baseline source is `experiments/midnight/` (mirror of submitted Midnight PR artifacts).
- Test registry is `experiments/midnight_lab/tests.tsv`.
- Runner is `experiments/midnight_lab/run_tests.sh`.
- No Smart U-Net paths are used here.

## Commands

List tests:

```bash
bash experiments/midnight_lab/run_tests.sh list
```

Run one test:

```bash
SEED=444 NPROC_PER_NODE=4 COMPILE_ENABLED=0 \
bash experiments/midnight_lab/run_tests.sh run midnight_control_screen
```

Run priority batch:

```bash
SEED=444 NPROC_PER_NODE=4 COMPILE_ENABLED=0 \
bash experiments/midnight_lab/run_tests.sh batch 1
```

Logs and summary:
- `experiments/midnight_lab/logs/runs/<run_stamp>/`
- `summary.tsv` includes status, proxy/final metrics, size, and log path.
