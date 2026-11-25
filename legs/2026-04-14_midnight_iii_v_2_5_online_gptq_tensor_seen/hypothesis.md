# Hypothesis — 2026-04-14_midnight_iii_v_2_5_online_gptq_tensor_seen

Parent: legs/2026-04-14_midnight_iii_v_2_3_online_gptq_cpu/train_gpt.py

## One Variable
- Name: `OnlineGptqAccumulator.n_seen`
- New value: scalar float tensors updated with in-place `add_`
- Baseline value: Python `int` counters (`self.n_seen[name] += rows`)

## Why
- `tracked_env.sh` matches the parent leg, including `GPTQ_CAL_MODE=online_cpu`.
- The trainer change stores each online-GPTQ row count as a tensor on the same device as the Hessian instead of mutating a Python int during the forward.

## Gate Pass Criteria
- `gate.sh` launches this leg's `train_gpt.py`, not the parent leg.
- Full-run logs, when produced, show `gptq:online mode:online_cpu` and a non-empty online calibration summary.
