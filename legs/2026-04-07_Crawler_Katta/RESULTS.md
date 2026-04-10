# Results: Crawler_Katta

Date: 2026-04-07 to 2026-04-08
Track: crawler
Parent: legs/2026-04-07_BW22_LoopDepth_9F/

## Verdict

[X] DOES NOT PROMOTE — "yup. it sucked"

## 8×H100 Gate (2000 steps, seed=444)

| Arm | Solver | Loops | raw BPB | int6_sw BPB | step_ms | Bytes | vs ctrl |
|-----|--------|-------|---------|-------------|---------|-------|---------|
| **A0_ctrl** | **Euler** | **3** | **1.2649** | **1.24486** | **110.22** | **15,630,277** | **—** |
| A1_rk2_fast | RK2 fast | 2 | 1.2672 | 1.24672 | 102.11 | 15,522,652 | +0.00186 |
| A2_rk4_fast | RK4 fast | 2 | CRASHED | — | — | — | — |
| A3_rk24_hybrid | RK2/4 | 2 | CRASHED | — | — | — | — |
| A4_rk24_hybrid | RK2/4 | 3 | CRASHED | — | — | — | — |

A2-A4 crashed: train_gpt_brotli.py line 2309 error in RK solver forward pass.

## 1×GPU Medium Sweep (3000 steps, seed=444)

| Arm | Solver | Loops | raw BPB | int6_sw BPB | step_ms | Bytes |
|-----|--------|-------|---------|-------------|---------|-------|
| M0_ctrl | Euler | 3 | 1.2647 | 1.25050 | 383.09 | 13,291,917 |
| M1_rk2 | RK2 fast | 2 | CRASHED | — | — | — |

## What we learned

- RK2 at 2 loops is 7.4% faster (102ms vs 110ms) but quality regresses +0.00186 BPB
- Throughput gain does NOT compensate: at 600s wallclock, ~428 extra steps, but quality loss wipes it out
- Higher-order RK solvers (RK4, hybrid) crashed — implementation bugs in solver forward pass
- Euler is already optimal for 3-loop crawler. RK only helps at high loop counts where numerical integration accuracy matters, but loops>3 is already net-negative from quant gap compounding
- DEAD. Do not revisit.

## 4-hour run recommendation

None. Dead concept.

## Data recovered

Results recovered 2026-04-10 from Codex agent session `019d69fd-b4e9-7fc2-8320-d816a77d72c8` in `~/.codex/history.jsonl`. Original pod logs destroyed with pod.
