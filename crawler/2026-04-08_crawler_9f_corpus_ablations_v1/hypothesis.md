# Hypothesis: crawler_9f_corpus_ablations_v1

Date: 2026-04-08
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Objective

Comprehensive screening ablation of every actionable improvement vector on the
BWX 9F base. The crawler battery is currently half-baked: TAP off, Anchor off,
only 3 undifferentiated loops, no QAT, naive uniform int6 quant. This screen
tests all known levers in a single economical pass to identify which ones carry
real signal on the 9F architecture before committing to expensive stacking runs.

**Environment:** 4xGPU, 1500 steps, seed=444
**Purpose:** Signal screen (not promotion gate). Winners get promoted to 2000-step 8x gate.

## Arms

### Battery / Recurrence (training runs, one variable each vs A00 control)

| Arm | Variable | Config Delta vs Control | Prior Evidence | Rationale |
|-----|----------|------------------------|----------------|-----------|
| A00 | Control | BWX 9F production config | 1.13868 BPB (full run) | Baseline |
| A01 | TAP shared | CRAWLER_TAP_DIM=32, CRAWLER_TAP_LOOP_SPECIFIC=0 | -0.00352 on BW5 (never tested on 9F) | Inter-loop reading: shared encoder tap gives loops a stable pre-quant anchor |
| A02 | Anchor | ANCHOR_DIM=32 | -0.00334 on BW5 (never tested on 9F) | Inter-loop writing: each loop commits compact state for the next loop |
| A03 | 4 loops naive | CRAWLER_LOOPS=4, CRAWLER_LOOP_ROPE_SCALES=9,1,1,1 | BW22 staged | +1 loop depth, repeat local RoPE scale |
| A04 | 4 loops differentiated | CRAWLER_LOOPS=4, CRAWLER_LOOP_ROPE_SCALES=9,3,1,1 | BW22 staged | +1 loop depth, differentiated battery |
| A05 | 5 loops progressive | CRAWLER_LOOPS=5, CRAWLER_LOOP_ROPE_SCALES=9,5,3,1,1 | BW22 A4; A3 showed -0.00261 | Deep progressive battery, strongest differentiation |
| A06 | Wider INST | INST_DIM=64 | Untested on 9F | Wider instruction bottleneck per loop (32 -> 64) |
| A07 | 2 crawler layers | NUM_CRAWLER_LAYERS=2 | Untested | Add a second crawler layer to the battery stack |
| A08 | Crawler int8 | CRAWLER_QUANT_INT8=1 | Failed in Ouro stack, never solo on 9F | Int8 quant for shared crawler block (multi-context resilience) |

### QAT Surrogates (training runs, one variable each vs A00 control)

| Arm | Variable | Config Delta vs Control | Prior Evidence | Rationale |
|-----|----------|------------------------|----------------|-----------|
| A09 | QAT legacy | QAT_ENABLED=1, QAT_SURROGATE=legacy | BW23 staged | STE QAT baseline — does QAT help at all on 9F? |
| A10 | QAT softclamp | QAT_ENABLED=1, QAT_SURROGATE=softclamp | BW23 staged, concept score 31 | Continuous surrogate: tanh-based smooth quantizer |
| A11 | QAT sigmoidste | QAT_ENABLED=1, QAT_SURROGATE=sigmoidste | BW23 staged, concept score 31 | Smooth staircase: sigmoid STE with beta=6.0 |

### Quant Policy (post-train on best checkpoint, no retrain)

| Arm | Variable | Config Delta vs Q00 | Rationale |
|-----|----------|---------------------|-----------|
| Q00 | Control export | INT6_CATS=mlp,attn,aux | Existing uniform policy |
| Q01 | Sensitivity: drop aux | INT6_CATS=mlp,attn | Promote aux params to int8, save quant error |
| Q02 | Aggressive: attn-only | INT6_CATS=attn | Only attention gets int6, rest higher precision |
| Q03 | Aggressive: mlp-only | INT6_CATS=mlp | Only MLP gets int6, rest higher precision |

## One-variable discipline

Each training arm changes exactly ONE config delta vs A00 control. All other
env vars are identical to BWX 9F production. Quant arms change only INT6_CATS
on a frozen checkpoint.

## Gate target

- **Screen pass:** any arm with delta_vs_ctrl <= -0.001 int6_sw_bpb at 1500 steps
- **Strong signal:** delta <= -0.002
- **Step_ms recorded per arm** for wallclock-budget impact assessment
- **Artifact bytes recorded** for 16MB cap compliance

## What this does NOT test (needs code first)

- LEVEL_SIGNAL_DIM (Ringformer-style loop conditioning) — not yet implemented
- DYNAMIC_LOOP_HALTING (difficulty-aware compute) — not yet implemented
- Persistent memory bridge — not yet implemented

These are next after this screen identifies which battery/QAT vectors carry signal.
