## Smart U-Net Test (Isolated)

This folder is an isolated test harness for Smart U-Net quality-based skip behavior.
It is copied from `midnight_v.2` so baseline files are not overwritten.

- Baseline untouched: `experiments/2026-04-07_midnight_v.2/train_gpt.py`
- Test script: `tests_smart_unet/train_gpt_smart_unet_test.py`
- Lane matrix: `tests_smart_unet/ablation_matrix.tsv`
- One-lane runner: `tests_smart_unet/run_ablation.sh`
- Batch runner: `tests_smart_unet/run_mode_batch.sh`

Routing modes:

- `soft_gating`: scales skip-link strength by batch quality (no layer removal).
- `hard_routing`: removes low-priority decoder layers under low quality.
- `competitive_routing`: decoder layers compete by learned skip importance under a quality budget.

Low-cost screen command:

```bash
SEED=444 NPROC_PER_NODE=8 PROFILE=screen PRIORITY_MAX=1 BUDGET_MINUTES=45 \
bash experiments/2026-04-07_midnight_v.2/tests_smart_unet/run_smart_unet_screen.sh
```
