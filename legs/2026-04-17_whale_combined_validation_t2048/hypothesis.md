# Hypothesis - whale_combined_validation_t2048

`fact`: Leg A selected `DKDV=128,128,8,3` and Leg B selected `DQ_TILE=128,64,8,5`, `DQ_MAXNREG=224` at `(2,2048,8,4,64)`.

`hypothesis`: Stacking those two winners in the heavier `10x200` confirmation pass should keep `whale_fast` under `fa3` on fwd+bwd wall-time at the same shape.

`proposal`: Run `whale_fast` versus `fa3` at the fixed competition-shape proxy `(2,2048,8,4,64)` with `--rounds 10 --iters 200`, keeping only the proven DKDV and DQ winners in tracked files.
