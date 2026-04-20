# Hypothesis - whale_cross_gpu_validation_prep_t2048

`fact`: The single-GPU lead frontier shifted to Hybrid B: `DQ=128,64,8,5`, `DKDV=128,128,8,3`, `DKDV_MAXNREG=256`, with best known exact-shape heavy point `QMR=1664`, whale `0.2856340484172106`, FA3 `0.2983246239423752`.

`fact`: The final winner-lock passes kept `1664` low enough to bless it as the carry-forward single-GPU anchor even though nearby `1952/2192/2240` remain valid same-family challengers.

`fact`: Previous wins only stayed trustworthy when each bench owned its own Triton cache and the launch path stayed one serial run per GPU.

`hypothesis`: A cross-GPU validation leg that reads a blessed single-GPU winner anchor, materializes probe queues from that anchor plus named challengers, isolates Triton/Inductor caches per worker, and promotes only green probes into heavy rechecks will let us confirm distributed signal quickly without reopening single-GPU search by default.

`proposal`: Treat Hybrid B `QMR=1664` as the current carry-forward anchor for downstream work, keep the scripts anchor-driven, and validate the anchor plus the strongest same-family and family-level challengers without rewriting the launch path again.
