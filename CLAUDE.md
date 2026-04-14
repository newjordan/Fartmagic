# Repo Rules

These rules apply to Claude, Codex, and any other coding agent working here.

## Scope
- Work only in `legs/<date>_<name>/` unless the user explicitly asks for framework work.
- Never edit `vault/` directly.
- Treat `vault/train_gpt_midnight_12l_sota_REAL.py` as frozen.
- Default parent for new work is `vault/train_gpt_midnight_iii_base.py`.

## Frozen: Pod Setup Script
- **`scripts/Im_sorry_pod_setup.sh` is FROZEN.  Do not edit without explicit user approval.**
- A frozen backup exists at `scripts/Im_sorry_pod_setup.sh.cu130_frozen_20260414`.
- The pod runs **CUDA 13.0 / PyTorch 2.11.0+cu130** on `vastai/pytorch:cuda-13.0.2-auto`.
- The FA3 wheel URL must target **cu130**.  Not cu124. Not cu128. cu130.
- The `/venv/main/bin` PATH prepend and the symlink fallback in `install_fa3()` are load-bearing.
- If you believe a change is needed, show evidence first, then ask the user.
- `pod_stack.lock` enforces the hash — if the guard fails, the setup script will not run.

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

## Startup Safety Protocol
- Before giving any paid run command, declare the canonical runtime repo path and branch.
- Never assume local path parity with pod paths; verify repo identity before run instructions.
- Required preflight for any gate/full run (evidence must be shown first):
  1. `pwd`
  2. `git remote -v`
  3. `git rev-parse --abbrev-ref HEAD`
  4. `test -f legs/<leg>/run.sh` (or `gate.sh`)
  5. `test -f <required tokenizer/model path>`
- If any preflight check fails, do not issue launch commands.
- If a startup error occurs, stop immediately. Do not issue a second launch until:
  1. one-line `fact` cause is named from the error output
  2. the fix is tracked in repo files (leg files/scripts), not shell-only
  3. updated launch command is provided with corrected path/repo context
- For compile-sensitive legs, compile behavior must be explicit in tracked files (`tracked_env.sh`), not implied defaults.
