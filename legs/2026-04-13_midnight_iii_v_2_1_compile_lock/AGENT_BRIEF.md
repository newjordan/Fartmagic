# Agent Brief — 2026-04-13_midnight_iii_v_2_postema_gptq

## Scope
You may edit only files listed in:
`legs/2026-04-13_midnight_iii_v_2_postema_gptq/agent_allowlist.txt`

## Task rules
1. Change exactly ONE variable vs baseline.
2. Do not edit `vault/`, `records/`, or `LOCKED_SOTA/`.
3. This leg is an isolated test silo. If you need a different test condition, create a new leg.
4. Update `hypothesis.md` before code edits.
5. Edit `tracked_env.sh` for env-level changes. Do not test by typing env overrides into the shell.
6. Run `python3 scripts/leg_diff_guard.py legs/2026-04-13_midnight_iii_v_2_postema_gptq` before any gate, full run, or commit.
7. Treat diff guard FAIL as a blocker unless the user explicitly approved wider thresholds.
8. Fill `ablation.md` and `RESULTS.md` after each run.
9. Do not make log claims without corpus evidence (path + exact lines or exact grep/sed output).
10. Do not make code-history claims without git evidence (commit IDs or diff output).
11. Use `checked`, `verified`, `saved`, `logged`, `validated`, `root cause`, and `ready` only with a named supporting artifact.
12. On evidence-heavy tasks, provide evidence first; for log-audit tasks, first output is corpus summary.
13. Label each statement as `fact`, `inference`, or `proposal`.
14. Treat plausible-but-unverified as failure.

## Commands
```bash
bash scripts/agent_guard.sh snapshot pre_midnight_iii_v_2_postema_gptq
python3 scripts/leg_diff_guard.py legs/2026-04-13_midnight_iii_v_2_postema_gptq
bash legs/2026-04-13_midnight_iii_v_2_postema_gptq/gate.sh
bash legs/2026-04-13_midnight_iii_v_2_postema_gptq/run.sh
bash scripts/agent_guard.sh check legs/2026-04-13_midnight_iii_v_2_postema_gptq/agent_allowlist.txt --delta-latest
```
