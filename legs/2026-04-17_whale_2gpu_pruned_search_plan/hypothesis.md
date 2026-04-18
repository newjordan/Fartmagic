# Hypothesis - whale_2gpu_pruned_search_plan

`fact`: Wave 1 selected `DQ_TILE=128,64,8,2` and `DQ_MAXNREG=160` from `worker0_dqtile_128_64_8_2_dqmr_160_dkdvmr_256_20260418_010329.json`, and the promotion log `logs/wave2_select_20260418_010948.log` recorded that choice.

`fact`: On the promoted DQ anchor, `DKDV_MAXNREG=160`, `192`, and `224` all lost to FA3, while `DKDV_MAXNREG=256` won in `evidence/wave2_dkdvmr_256_dq_128_64_8_2_dqmr_160_20260418_011808.json`.

`fact`: Heavy `10x200` validations on the `DQ_TILE=128,64,8,2` ridge stayed green at `DQ_MAXNREG in {160,192,224,256}`, with the best current single result at `DQ_MAXNREG=256` in `evidence/manual_validate_dq_128_64_8_2_dqmr_256_dkdvmr_256_20260418_012119.json`.

`fact`: `DQ_TILE=128,64,8,5` remains a live fallback family; `DQ_MAXNREG=224` stayed green in `evidence/manual_validate_dq_128_64_8_5_dqmr_224_dkdvmr_256_20260418_012031.json`.

`hypothesis`: The T=2048 crossover is now real on this stack, and the remaining search is a narrow ranking problem: keep `DKDV_MAXNREG=256` fixed, continue only the high-side DQ maxnreg ridge, and verify repeatability between the nearby green points.

`proposal`: Preserve `DQ_TILE=128,64,8,2`, `DQ_MAXNREG=256`, `DKDV_TILE=128,128,8,3`, `DKDV_MAXNREG=256` as the current anchor, then run only one-variable follow-ups above `DQ_MAXNREG=256` or repeatability reruns on `224` vs `256`.
