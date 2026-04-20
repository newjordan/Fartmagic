# Hypothesis - whale_cross_gpu_validation_prep_t2048

`fact`: The single-GPU lead frontier has shifted to Hybrid B: `DQ=128,64,8,5`, `DKDV=128,128,8,3`, `DKDV_MAXNREG=256`, with current best known exact-shape heavy point `QMR=1664`, whale `0.2856340484172106`, FA3 `0.2983246239423752`.

`fact`: The current lead family is broad enough that downstream readiness should read from an anchor file, not from hardcoded per-script constants.

`fact`: Previous wins only stayed trustworthy when each bench owned its own Triton cache and the launch path stayed one serial run per GPU.

`hypothesis`: A cross-GPU validation leg that reads a provisional winner anchor, materializes probe queues from that anchor plus named challengers, isolates Triton/Inductor caches per worker, and promotes only green probes into heavy rechecks will let us confirm distributed signal quickly once the single-GPU carry-forward family is stable enough.

`proposal`: Prepare a launch-ready leg now, but keep it dormant until the single-GPU winner is promoted beyond `status=provisional`. When needed, this leg should validate the current anchor family plus the strongest challenger families without requiring another script rewrite.
