# SOTA_FINAL

Minimal isolated repo for the Midnight neural line.

What lives here:
- `vault/train_gpt_midnight_12l_sota_REAL.py`: frozen official leader
- `vault/train_gpt_midnight_iii_base.py`: working base for new growth experiments
- `scripts/pod_setup.sh`: single blessed pod bootstrap
- `scripts/new_leg.sh`: create a tracked experiment folder
- `scripts/leg_diff_guard.py`: block overly broad trainer edits
- `scripts/agent_guard.sh`: block out-of-scope file drift
- `legs/2026-04-12_midnight_12l_clean`: current 12L SP8192 campaign
- `legs/2026-04-12_midnight_iii_clean`: current III SP8192 campaign

Operating rules:
- Do not run ad-hoc experiment env changes directly against `vault/` or a leg trainer from the shell.
- Every experiment gets its own `legs/<date>_<name>/` folder with tracked files.
- New work starts from `vault/train_gpt_midnight_iii_base.py` unless you explicitly need the frozen 12L leader.
- Use `bash scripts/new_leg.sh <name>` to create a new leg.
- Run only a leg's tracked `gate.sh` or `run.sh`.

Fast start:
```bash
bash scripts/pod_setup.sh
bash scripts/new_leg.sh qk_gain_probe
bash legs/2026-04-12_midnight_iii_clean/gate.sh
```

Reference map: see `PIPELINE.md` and `RESEARCH_LINKS.md`.
