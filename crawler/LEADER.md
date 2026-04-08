# Crawler SOTA — Current Leader

Score:  1.13867894 BPB (seed 444) | mean pending (seed 300 not recorded in-tree)
Size:   15,239,617 bytes (15.24MB, seed 444) | seed 300 pending
Date:   2026-04-02
Leg:    records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/
Run:    SEED=444 NPROC_PER_NODE=8 bash records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/run.sh

## Architecture
Bandit_Wagon_X_9F (tap-off crawler stack)
NUM_FLAT_LAYERS=9 | NUM_CRAWLER_LAYERS=1 | CRAWLER_LOOPS=3 | INST_DIM=32
CRAWLER_TAP_DIM=0 | ANCHOR_DIM=0 | COMPILE_FULLGRAPH=1 | SKIP_GPTQ=1
CRAWLER_LOOP_ROPE_SCALES=9,1,1 | 110.19ms/step on 8xH100 | 5446 steps in 600s

## Seeds
| Seed | BPB exact       | Size       | Status                               |
|------|-----------------|------------|--------------------------------------|
| 444  | 1.13867894      | 15,239,617 | current best confirmed in-tree full run |
| 300  | pending         | pending    | confirmation run not recorded in-tree |
| mean | pending         | --         | requires seed 300                    |

Reference metrics source:
`crawler/2026-04-02_Research_Audit_24h/RESULTS_AUDIT_2026-04-01_to_2026-04-02.md`

Ouroboros note:
PR #1283/#1308 reports stronger results on a separate submission lineage.
Those record folders are not present in current `TEST_LAB` tree, so BWX remains the in-tree promotion baseline until that lineage is imported/reconciled.

## Promotion Gate
Beat 1.13867894 on seed 444 with artifact <= 16,000,000 bytes → confirm on seed 300 → update this file.
One variable changed per leg. Gate (1-GPU, 2000 steps) before any 8x run.
