# 2-Worker Pruned Bandit Search

## Objective

Use the assigned 2x H100 node to search only the high-signal interaction basin around the best clean scratch-controlled stack at `(2,2048,8,4,64)` and preserve any whale `<` FA3 crossover with committed evidence.

## Frozen kernel source

- Scratch kernel copied from commit `16cf4c5`
- No edits to `vault/`
- Record kernel sha256 in every worker log and JSON artifact

## Executed wave 1

- Fixed `DKDV_TILE=128,128,8,3` and `DKDV_MAXNREG=256`
- Worker 0 screened `DQ_TILE in {128,64,8,2 128,64,8,3}` crossed with `DQ_MAXNREG in {128 160 192 224}`
- Worker 1 screened `DQ_TILE in {128,64,8,4 128,64,8,5}` crossed with `DQ_MAXNREG in {128 160 192 224}`
- Promotion record: `logs/wave2_select_20260418_010948.log`

## Executed wave 2

- Heavy `10x200` validation on promoted DQ points with `DKDV_TILE=128,128,8,3` and `DKDV_MAXNREG=256`
- Light `5x100` DKDV maxnreg sweep over `160 192 224 256` on the promoted DQ anchor
- Follow-up heavy confirmations on neighboring DQ maxnreg points and one alternate DQ family

## Current anchor

- `DKDV_TILE=128,128,8,3`
- `DKDV_MAXNREG=256`
- `DQ_TILE=128,64,8,2`
- `DQ_MAXNREG=256`
- Best current heavy result: whale `0.298562224105 ms` vs FA3 `0.318293856025 ms` in `evidence/manual_validate_dq_128_64_8_2_dqmr_256_dkdvmr_256_20260418_012119.json`

## Launch

- Use `launch_remote.sh HOST PORT` for the initial 2-worker screen.
- Use `run_single_case.sh` and `chain_after_dq_2gpu.sh` for wave-2 promotion, heavy validation, and follow-up ridge checks.
