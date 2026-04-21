# Hypothesis - whale_cross_gpu_validation_prep_t2048 — 2026-04-21

`fact`: The single-GPU lead frontier shifted to Hybrid B: `DQ=128,64,8,5`, `DKDV=128,128,8,3`, `DKDV_MAXNREG=256`, with best known exact-shape heavy point `QMR=1664`, whale `0.2856340484172106`, FA3 `0.2983246239423752`.

`fact`: The final winner-lock passes kept `1664` low enough to bless it as the carry-forward single-GPU anchor even though nearby `1952/2192/2240` remain valid same-family challengers.

`fact`: Previous wins only stayed trustworthy when each bench owned its own Triton cache and the launch path stayed one serial run per GPU.

`fact`: The completed `2x` validation reordered the winner. The final head-to-head replay selected primary `DQ=128,64,8,5 / DKDV=128,64,8,3 / DKDV_MAXNREG=256 / QMR=1984` over fallback `QMR=928`.

`hypothesis`: Once the carry-forward anchor is promoted from single-GPU to `2x`, the next meaningful contradiction can only come from donor integration or end-to-end training pressure, not from reopening routine kernel search.

`proposal`: Treat primary `QMR=1984` as the current carry-forward anchor for donor work, preserve fallback `QMR=928` as the strongest displaced challenger, and keep this leg as evidence rather than as an open frontier.
