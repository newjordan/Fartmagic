# Repo Rules

These rules apply to Claude, Codex, and any other coding agent working here.

## Scope
- Work only in `legs/<date>_<name>/` unless the user explicitly asks for framework work.
- Never edit `vault/` directly.
- Treat `vault/train_gpt_midnight_12l_sota_REAL.py` as frozen.
- Default parent for new work is `vault/train_gpt_midnight_iii_base.py`.

## Experiment Discipline
- Do not test ideas by typing env overrides directly into the shell.
- Put experiment changes in tracked leg files.
- If the change is an env-level experiment, edit `legs/<leg>/tracked_env.sh`.
- If the change is a code-level experiment, edit `legs/<leg>/train_gpt.py`.
- Keep one variable per leg unless the user explicitly approves a wider change.

## Required Flow
1. Update `hypothesis.md`
2. Run `python3 scripts/leg_diff_guard.py legs/<leg>`
3. Run `bash legs/<leg>/gate.sh` or `bash legs/<leg>/run.sh`
4. Update `ablation.md` and `RESULTS.md`

## Hard Stops
- If a desired change is not tracked in repo files, stop and add it to the leg first.
- If the diff guard fails, stop unless the user explicitly approves wider thresholds.
- If you need a new experiment, create it with `bash scripts/new_leg.sh <name>`.
