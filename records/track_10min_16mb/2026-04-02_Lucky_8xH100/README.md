## Lucky

Rascal II + SLOT: 8-step sliding-window test-time adaptation (lr=0.005) at inference. All other architecture unchanged from Rascal II.

## Results

| Seed | val_bpb (sliding window) | Steps | Size |
|------|--------------------------|-------|------|
| 444  | PENDING                  | PENDING | PENDING |
| 300  | PENDING                  | PENDING | PENDING |
| **mean** | **PENDING**          |       | **PENDING** |

Hardware: 8xH100 SXM · 600s wallclock · `bytes_code`: 123854

## Architecture changes

- Added SLOT: 8-step sliding-window test-time adaptation at inference (lr=0.005)

## Reproduce

```bash
SLOT_ENABLED=1 SKIP_GPTQ=1 SEED=444 python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-02_Lucky_8xH100/train_gpt.py
```
