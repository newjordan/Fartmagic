# Agent Brief — 2026-04-12_midnight_iii_clean

## Scope
You may edit only files listed in:
`legs/2026-04-12_midnight_iii_clean/agent_allowlist.txt`

## Task rules
1. Change exactly ONE variable vs baseline.
2. Do not edit `vault/`, `records/`, or `LOCKED_SOTA/`.
3. Update `hypothesis.md` before code edits.
4. Edit `tracked_env.sh` for env-level changes. Do not test by typing env overrides into the shell.
5. Run `python3 scripts/leg_diff_guard.py legs/2026-04-12_midnight_iii_clean` before any gate, full run, or commit.
6. Treat diff guard FAIL as a blocker unless the user explicitly approved wider thresholds.
7. Fill `ablation.md` and `RESULTS.md` after each run.

## Commands
```bash
bash scripts/agent_guard.sh snapshot pre_midnight_iii_clean
python3 scripts/leg_diff_guard.py legs/2026-04-12_midnight_iii_clean
bash legs/2026-04-12_midnight_iii_clean/gate.sh
bash legs/2026-04-12_midnight_iii_clean/run.sh
bash scripts/agent_guard.sh check legs/2026-04-12_midnight_iii_clean/agent_allowlist.txt --delta-latest
```
