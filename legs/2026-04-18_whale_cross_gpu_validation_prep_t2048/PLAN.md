# Cross-GPU Plan

Goal: keep multi-GPU validation launch-ready while single-GPU kernel work remains primary. Do not launch this leg yet. It exists so we can swap directly into distributed signal checks once the single-GPU carry-forward family is stable enough.

Anchor model:
1. `winner_anchor.json` is the single source of truth for the provisional carry-forward family.
2. The launch scripts read the anchor at runtime instead of hardcoding the winner in multiple places.
3. The anchor can remain `status=provisional` until the single-GPU frontier is fully bracketed.
4. Downstream scripts must treat the anchor as a live carry-forward spec, not as a final locked winner.

Execution shape:
1. Run `prepare_cross_gpu_validation.sh` to materialize a probe queue from the current anchor plus named challenger families.
2. Run `launch_cross_gpu_validation.sh` on a Vast host with isolated workers, one serial bench per GPU.
3. Keep each worker on its own `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and `XDG_CACHE_HOME`.
4. Promote any green probe into a heavier validation pass immediately instead of waiting for a fresh manual launch.
5. Use `verify_cross_gpu_launch.sh` to check PID, GPU utilization, anchor state, log tail, and evidence growth within 60 seconds of launch.

Coverage target:
- Anchor family from `winner_anchor.json`
- Direct challengers kept ready behind the anchor:
  - primary `DQ=128,64,8,5 / DKDV=128,64,8,3`
  - fallback `DQ=128,64,8,2 / DKDV=128,128,8,3`
  - hybrid A `DQ=128,64,8,2 / DKDV=128,64,8,3`
- QMR bands centered on the current winning floor and the strongest challenger checkpoints

Evidence rules:
- Every claim must be backed by a log line, JSON artifact, or file path.
- Do not collapse green probes and heavy validations into the same label.
- Do not share a cache directory between workers.
- Do not call the anchor final until its status is explicitly promoted.
