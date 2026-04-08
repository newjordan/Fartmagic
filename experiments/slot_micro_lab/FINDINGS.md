# SLOT Micro Lab Findings

Run:

```bash
bash experiments/slot_micro_lab/run_micro.sh
```

Observed results:

| case | raw bytes | compressed bytes | outcome |
|---|---:|---:|---|
| `ctrl` | 133789 | 121406 | baseline |
| `slot_helper_unused` | 133789 | 121406 | size-neutral |
| `slot_post_export` | 133789 | 121406 | size-neutral |
| `contam_state_dict` | 134233 | 121579 | grows |
| `contam_pre_export` | 133789 | 121409 | effectively size-neutral on this toy model |

Exact mechanism:

- A SLOT helper that exists but is never invoked on the export path does not change serialized bytes.
- A SLOT helper invoked only after export does not change serialized bytes.
- The only direct byte growth in this micro-harness came from adding a new tensor to the exported `state_dict`.
- A persistent tensor mutation before export did not change the raw serialized size here because tensor shape and key set stayed fixed; it only nudged compressed size by a few bytes.

Conclusion:

- The size problem is not caused by SLOT logic existing in isolation.
- The live risk is SLOT-adjacent contamination of the exported state, not post-export review code.
- If SLOT is kept post-export and state-dict contents are unchanged, the blob stays size-neutral in this micro setup.
