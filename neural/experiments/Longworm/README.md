# Longworm Experiment

This folder is the dedicated Longworm experiment surface.

- Trainer: `train_longworm.py`
- Base lineage: copied from `experiments/Rascal_III/train_gpt.py` and then evolved here.
- Isolation rule: Longworm-only concepts must be implemented here, not in Rascal experiment files.

Transition-integrator knobs:

- `TRANSITION_INTEGRATOR=euler|rk2|rk4|hybrid_k2_k4`
- `TRANSITION_HYBRID_LAST_N=<int>` (used with `hybrid_k2_k4`)

Notes:

- `euler` preserves baseline behavior.
- `rk2` and `rk4` increase compute per layer for potentially better transition quality.
- `hybrid_k2_k4` runs RK2 on early layers and RK4 on the last `N` layers.
