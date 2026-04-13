# Repo Rules

These rules apply to Claude, Codex, and any other coding agent working here.

## Scope
- Work only in `legs/<date>_<name>/` unless the user explicitly asks for framework work.
- Never edit `vault/` directly.
- Treat `vault/train_gpt_midnight_12l_sota_REAL.py` as frozen.
- Default parent for new work is `vault/train_gpt_midnight_iii_base.py`.

## Experiment Discipline
- Every experiment/test must be a brand-new leg created with `bash scripts/new_leg.sh <name>`.
- Never repurpose an existing leg for a new hypothesis; create a child leg instead.
- Hardcode the tested condition into that leg's tracked files.
- Do not test ideas by typing env overrides directly into the shell.
- Put experiment changes in tracked leg files.
- If the change is an env-level experiment, edit `legs/<leg>/tracked_env.sh`.
- If the change is a code-level experiment, edit `legs/<leg>/train_gpt.py`.
- Keep one variable per leg unless the user explicitly approves a wider change.

## Evidence Discipline
- No claim about logs without corpus evidence.
- Any log claim must cite the extracted corpus path or raw transcript path.
- Any log claim must include exact line references or exact `grep`/`sed` output.
- No claim about code history without git evidence.
- Any code-history claim must cite commit IDs or `git diff` output.
- Do not use `checked`, `verified`, `saved`, `logged`, `validated`, `root cause`, or `ready` unless the supporting artifact is named.
- On evidence-heavy tasks, implementation is blocked until evidence is presented first.
- For log-audit tasks, first output must be a corpus summary, not a solution scaffold.
- Separate each statement as `fact`, `inference`, or `proposal`.
- Treat plausible-but-unverified as failure, not partial success.

## Required Flow
1. Create a new leg with `bash scripts/new_leg.sh <name>`.
2. Update that leg's `hypothesis.md`.
3. Encode the test in that leg's `tracked_env.sh` and/or `train_gpt.py`.
4. Run `python3 scripts/leg_diff_guard.py legs/<leg>`.
5. Run `bash legs/<leg>/gate.sh` or `bash legs/<leg>/run.sh`.
6. Update `ablation.md` and `RESULTS.md`.

## Hard Stops
- If a desired change is not tracked in repo files, stop and add it to the leg first.
- If a new idea requires changing test conditions, stop and create a new leg.
- If the diff guard fails, stop unless the user explicitly approves wider thresholds.
- If you need a new experiment, create it with `bash scripts/new_leg.sh <name>`.
- If evidence for a claim is missing or ambiguous, stop and mark the claim unsupported.
