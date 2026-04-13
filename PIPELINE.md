# Midnight Pipeline

## Current Legal SOTA
- Leg: `legs/2026-04-13_midnight_iii_v_qattn6`
- Status: promoted (submission-legal artifact)
- Best exact seed in corpus: `1.09159026` (`seed=444`)
- Size at promoted point: `15576180` bytes mixed+brotli
- Corpus evidence:
  - `legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed444_20260413_044438.log:111`
  - `legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed444_20260413_044438.log:107`

## Frozen Reference
- File: `vault/train_gpt_midnight_12l_sota_REAL.py`
- Status: frozen reference
- Prior documented best exact seed: `1.10567949` (`seed=444`)
- Role: legality-clean historical control

## Working Base
- File: `vault/train_gpt_midnight_iii_base.py`
- Status: active growth base
- Prior documented best exact seed: `1.10616680` (`seed=444`)
- Role: default parent for new legs

## Blessed Workflow
1. Run `bash scripts/pod_setup.sh`
2. Create a new tracked leg with `bash scripts/new_leg.sh <name>`
3. Update that leg's `hypothesis.md`
4. Hardcode the tested condition inside that leg's tracked files only
5. Run `python3 scripts/leg_diff_guard.py legs/<leg_name>`
6. Run the leg's `gate.sh` or `run.sh`
7. Fill `ablation.md` and `RESULTS.md`

## Hard Rules
- Do not edit `vault/` directly.
- Do not test new ideas with shell-typed env overrides against a leg or vault file.
- Every test is a new leg; do not mutate an old leg into a different experiment.
- Keep one variable per leg unless you explicitly re-baseline the branch.
- Use `scripts/agent_guard.sh` if an agent is touching the repo.

## Active Legs
- `legs/2026-04-12_midnight_12l_clean`
- `legs/2026-04-12_midnight_iii_clean`
- `legs/2026-04-13_midnight_iii_v_qattn6`

## Lane Runner Entrypoints
- `bash scripts/run_midnight_iii_lane.sh gate|run [seed] [nproc_per_node]`
- `bash scripts/run_midnight_12l_lane.sh gate|run [seed] [nproc_per_node]`
