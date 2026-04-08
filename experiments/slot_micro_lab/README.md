# SLOT Micro Lab

Tiny deterministic harness for checking whether SLOT-like additions change serialized model bytes.

Cases:
- `ctrl`
- `slot_helper_unused`
- `slot_post_export`
- `contam_state_dict`
- `contam_pre_export`

Run:

```bash
bash experiments/slot_micro_lab/run_micro.sh
```

Interpretation:
- `slot_helper_unused` and `slot_post_export` should match `ctrl`
- `contam_state_dict` should grow
- `contam_pre_export` may change bytes if a persistent tensor is mutated before export
