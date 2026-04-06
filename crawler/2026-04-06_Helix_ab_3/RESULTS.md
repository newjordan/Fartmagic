# Results: Helix_ab_3

Date: 2026-04-06
Track: crawler
Parent: legs/2026-03-29_BW5/

## Verdict

[ ] PROMOTES  [x] DOES NOT PROMOTE

## Gate Run

Command:
`SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-06_Helix_ab_3/gate.sh`

Summary file:
`/workspace/parameter-golf/crawler/2026-04-06_Helix_ab_3/results/summary_s444_20260406_194948.tsv`

## Gate Scores (seed=444)

| Arm | model_params | raw_bpb | int6_sw_bpb | step_ms | artifact_bytes | delta_vs_ctrl |
|-----|--------------|---------|-------------|---------|----------------|---------------|
| HAB3-00_ctrl | 14,462,508 | 1.3080 | 1.28905275 | 72.77 | 8,887,004 | 0.00000000 |
| HAB3-01_helix | 15,183,916 | 1.3351 | 1.42860043 | 86.35 | 9,320,278 | +0.13954768 |

Gate target: `delta < -0.003 int6_sw_bpb`

Observed: `+0.13954768` (hard fail).

## What we learned

- Helix SplitHead (dim=384, stride=1, cross=8, loops=1) regressed massively at 2000-step gate scale.
- Throughput regressed vs control (86.35ms vs 72.77ms), though still below the 100ms hard cap.
- Artifact sizes remained legal and compact; quality was the blocker.

## Next hypothesis

- Do not promote this Helix_ab_3 configuration to a 600s production run.
- Keep Ouroboros/BWX lineage as the active path.
