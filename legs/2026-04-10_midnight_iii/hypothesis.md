# Hypothesis — Midnight III

## Variables (stacked from individually screened arms)
- `QK_GAIN_INIT`: 1.5 → 5.25 (from qk_gain_525 screen)
- `MATRIX_LR`: 0.025 → 0.022 (from matrix_lr_022 screen, best single arm -0.0136 BPB)
- `MUON_WD`: 0.04 → 0.095 (from muon_wd_095 screen, -0.0021 BPB + faster steps)
- Parallel residual layers 7-11 (from parallel_residual screen, -0.0096 BPB)
- Depth recurrence: loop layers 3-5 x2, activate at 35% wallclock (from depth_recurrence screen)

## Parent
- Source: `vault/train_gpt_midnight_12l_sota_REAL.py`
- SHA256: `301022ac7b00f76557636efd1bda555aeb9ea34d345bfb0dc9c662cb7acdc6b5`

## Why
All five techniques come from the same synergistic lineage. Individual 2xH100
6-min screens showed:
- control: 1.3181
- matrix_lr_022: 1.3045 (-0.0136)
- parallel_residual: 1.3085 (-0.0096)
- muon_wd_095: 1.3160 (-0.0021)
- depth_recurrence: 1.3201 (+0.0020, but throughput-limited on short screen)
- stack_core (all combined): 1.3243 (+0.0062, depth recurrence dominated on 6-min)

Depth recurrence hurt on 6-min 2xH100 screens due to throughput loss, but is
expected to pay off on longer 600s 8xH100 runs where it has more steps after
activation and more parallelism to absorb the cost.

## Gate Target
Full 8xH100 600s run. Beat current leader 1.10567949 BPB on seed 444.
