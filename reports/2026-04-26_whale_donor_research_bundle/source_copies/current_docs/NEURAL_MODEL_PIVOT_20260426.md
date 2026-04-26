# Neural Model Pivot - 2026-04-26

## Direction

The active objective is the best neural model under the contest constraints.
Eval-time tricks, post-train adaptation, sliding eval, byte polish, and stack
parity work are secondary packaging. They do not define the research backbone.

## Corrections

- The PR1812 / cu128-era field-stack reproduction path is demoted. Do not spend
  more work chasing that stack unless the user explicitly asks for a runtime
  parity investigation.
- A result that only beats the old PR1493 reference through eval tricks is not
  the target. The target is a stronger learned model body.
- PR1493 is historical evidence only until the neural-body candidate screen is
  complete. Do not launch, recommend, or center another PR1493-family run before
  the 17 candidates are tested.
- The failed PR1812 4x run and the failed PR1493-gated probe remain archived
  evidence. They are not the forward direction.

## Active Gate

First test the 17 neural-body candidates. The first-pass read should be raw /
post-EMA neural quality under comparable mechanics. Compression, TTT, sliding,
and final byte shaping are only promoted after a body shows it is worth
packaging.

## Immediate Work Order

1. Materialize the 17-candidate slate in a manifest before any new launch.
2. Check each candidate has a single dated test-car `run.py` source of truth.
3. Run candidates under comparable wallclock, data, evaluator, seed policy, and
   hardware shape.
4. Promote only candidates with a real neural-body signal.
5. After the screen completes, choose the best body for quantization and final
   eval packaging.

## Stop Rules

- No stack-parity rabbit hole.
- No eval-trick-first leaderboard chase.
- No old PR1493 anchor rerun or PR1493-family detour until the 17-candidate
  screen is complete.
- No launch file that is not a dated test car.
