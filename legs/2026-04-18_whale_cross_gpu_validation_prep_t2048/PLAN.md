# Cross-GPU Plan — 2026-04-21

Goal: preserve the finished `2x` validation record and hand a single blessed distributed carry-forward into donor integration without rewriting the winner spec again.

Anchor model:
1. `winner_anchor.json` is the single source of truth for the blessed downstream carry-forward family.
2. The launch scripts read the anchor at runtime instead of hardcoding the winner in multiple places.
3. The anchor is now promoted beyond `status=single_gpu_blessed`; downstream scripts should treat it as the default post-`2x` launch spec.
4. Single-GPU and displaced `2x` challengers still matter as evidence, but they no longer control the active carry-forward unless donor integration reopens the question.

Execution outcome:
1. The `2x` gate was run to completion with isolated workers, one serial bench per GPU.
2. The final winner-lock block reduced the field to:
   - fallback `928`
   - primary `1984`
3. Primary `1984` beat fallback `928` in the final head-to-head replay pair and became the `2x` carry-forward.
4. This leg now serves as the evidence base and anchor source for donor integration rather than an active open search.

Coverage outcome:
- Single-GPU blessed anchor was tested and challenged under `2x` pressure.
- Final `2x` carry-forward:
  - primary `DQ=128,64,8,5 / DKDV=128,64,8,3 / DKDV_MAXNREG=256 / QMR=1984`
- Strongest displaced `2x` challenger:
  - fallback `DQ=128,64,8,2 / DKDV=128,128,8,3 / DKDV_MAXNREG=256 / QMR=928`
- Secondary and wider-family challengers are archived as evidence, not active frontier work.

Evidence rules:
- Every claim must be backed by a log line, JSON artifact, or file path.
- Do not collapse green probes and heavy validations into the same label.
- Do not share a cache directory between workers.
- The current anchor is now promoted for donor use; if it changes again, that change should be driven by donor contradiction or end-to-end pressure, not by default kernel drift.
