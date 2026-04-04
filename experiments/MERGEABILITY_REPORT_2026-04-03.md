# Mergeability Report (April 3, 2026)

## Executive Summary

Your research signal is strong. Your current mergeability is being limited mostly by PR hygiene, not model quality.

Primary issue: current open PRs appear to come from a very large rolling branch, producing huge diffs that are hard to review and trust.

## Snapshot: Current Open PRs (public)

| PR | Title (short) | Status | Claimed mean BPB | Commits | Changed files | Mergeability risk |
|---|---|---|---:|---:|---:|---|
| #1120 | Rascal | Open | 1.1099 | 141 | 258 | Medium-High |
| #1208 | Nightcrawler | Open | 1.1761 | 306 | 1701 | High |
| #1282 | Slot Machine | Open | 1.1042 | 404 | 1797 | High |
| #1286 | Lucky IV | Open | 1.0964 | 410 | 1805 | High |
| #1308 | Ouroboros research | Open | 1.1364 | 412 | 1807 | High |

Interpretation:
- Performance claims are competitive.
- Review surface is too large for maintainers to safely merge quickly.

## Honest Optics Assessment

What looks strong:
- High experiment velocity.
- Clear originality (crawler line, cubric lineage, slot variants).
- Reproducibility intent is usually present (seeds/logs/commands).

What weakens merge optics:
- Many overlapping open PRs on the same rolling codebase.
- Very large non-isolated diffs (noise + accidental coupling risk).
- Superseded PRs left open can make the queue harder to triage.

## Mergeability Scorecard (0-10)

- Technical novelty: **9**
- Empirical output: **9**
- Reproducibility evidence: **8**
- Reviewer friendliness: **4**
- Branch hygiene / isolation: **3**
- Net mergeability today: **5.5**

If you isolate one candidate into a clean minimal PR, net mergeability can jump to ~**8+** quickly.

## Fast Path to “Looks Great”

1. Pick one flagship PR to carry right now.
- Recommended candidate: **#1286 (Lucky IV)** because score is strong and narrative is simple.

2. Freeze the story.
- Keep one record claim active.
- Convert others to supporting references or close as superseded.

3. Re-cut from clean `upstream/main`.
- New branch from fresh upstream.
- Add only one record folder and only files required by submission rules.
- No cross-folder collateral edits.

4. Keep diff tiny and deterministic.
- Target: one record directory + minimal metadata updates only.

5. Include a strict compliance block in README.
- 3-seed mean + std
- Training/eval wallclock
- Artifact bytes
- Explicit legality statement (score-first / no pre-eval adaptation)

6. Add one reviewer-facing “why trust this” section.
- Exact commands
- Expected outputs
- Log file names and seed mapping

## Recommended PR Queue Hygiene

Do this in order:

1. Keep active:
- `#1286` (if this is your current best practical contender)
- `#1308` (research/non-record track, if you want crawler story preserved)

2. Mark as superseded or close:
- `#1282` superseded by `#1286`
- Older crawler/rascal threads that are no longer your lead claim

3. Avoid parallel “competing primaries.”
- One primary record PR at a time improves reviewer confidence and response speed.

## Submission-Quality Checklist (must pass before asking for merge)

- [ ] 3 seeds, mean/std shown clearly
- [ ] `submission.json` complete and consistent with README/logs
- [ ] `train_gpt.py` and dependencies run from the record folder
- [ ] Artifact bytes reported and below limit
- [ ] Training <= 600s and eval <= 600s on 8xH100 documented
- [ ] No validation leakage / no pre-eval adaptation
- [ ] Repro command is one copy-paste block
- [ ] Diff contains only intended record files

## Suggested Next Move (Practical)

Run a clean “repack PR” workflow for the flagship:

1. Create clean branch from `upstream/main`.
2. Copy only the finalized record folder for the chosen run.
3. Validate file tree + local run command sanity.
4. Open fresh PR with a short, strict template.
5. Link prior PRs as lineage and mark superseded where appropriate.

This gives maintainers a low-risk merge target and preserves your technical momentum.

