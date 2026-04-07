# Hypothesis: BW22_LoopDepth_9F

Date: 2026-04-07
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## What changes

Sweep crawler loop depth (3, 4, 5) and battery RoPE scales on 9F base.
This is a multi-arm ablation to inform the 4-hour unlimited-compute showcase run
requested by competition organizers.

## Arms

| Arm | CRAWLER_LOOPS | CRAWLER_LOOP_ROPE_SCALES | Rationale |
|-----|--------------|--------------------------|-----------|
| A0  | 3            | 9,1,1                    | Control (BWX 9F production config) |
| A1  | 4            | 9,1,1,1                  | +1 loop, naive battery (repeat local) |
| A2  | 4            | 9,3,1,1                  | +1 loop, differentiated battery |
| A3  | 5            | 9,3,1,1,1                | +2 loops, differentiated battery |
| A4  | 5            | 9,5,3,1,1                | +2 loops, progressive battery |

## Why

Competition organizers specifically requested a 4-hour universal transformer /
depth recurrence run on the unlimited compute leaderboard. The crawler is
depth recurrence. More loops = deeper effective model using the same shared
weights.

Smokestack validated that each loop does genuine work (-0.004 to -0.005 BPB).
Reducing loops hurts. This tests whether ADDING loops helps, and whether
battery differentiation matters for loops 4 and 5.

At 4 hours: loops=4 gets ~98k steps, loops=5 gets ~78k steps (vs ~130k at
loops=3). The quality-vs-throughput tradeoff is the key question.

## Gate target

- int6_sw_bpb delta vs control: any negative delta justifies further testing
- step_avg: measure per-arm to estimate 4-hour step counts
- Compile must succeed with fullgraph=1 at all loop counts
- Model params: will increase slightly (more loop_inst_up projections)
