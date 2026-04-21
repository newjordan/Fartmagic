# Whale Kernel Pod Pull Summary — 2026-04-21

## Pulled archive

- Archive: `/home/frosty40/SOTA_FINAL/evidence/pod_pulls/2026-04-21_2x_winner_pull/whale_kernel_legs_2026-04-21.tar`
- Size: `634M`
- SHA256: `cd24792831889b2282156e882b9b02863225d8dffa9304ee628a599f3730dd43`

## Included legs

- `legs/2026-04-17_whale_2gpu_pruned_search_runtime`
- `legs/2026-04-18_whale_cross_gpu_validation_prep_t2048`

## Blessed single-GPU winner

- Family: `hybrid_b`
- Config: `DQ=128,64,8,5 / DKDV=128,128,8,3 / DKDV_MR=256 / QMR=1664`
- Best exact-shape heavy point:
  - whale `0.2856340484172106`
  - FA3 `0.2983246239423752`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_runtime/evidence/p250_morning_hybridA_vs_hybridB_10_g1_dq128_64_8_5_dkdv128_128_8_3_qmr1664_20260419_195209.json`
- Winner-lock support:
  - whale `0.28713868774473666`, FA3 `0.30372974422574045`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_runtime/evidence/p390_hybridB_final_bless_a_g0_dq128_64_8_5_dkdv128_128_8_3_qmr1664_20260420_214432.json`
  - whale `0.2902600641697645`, FA3 `0.3018856480717659`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-17_whale_2gpu_pruned_search_runtime/evidence/p393_hybridB_winner_lock_b_g0_dq128_64_8_5_dkdv128_128_8_3_qmr1664_20260420_223008.json`

## Blessed 2x winner

- Family: `primary`
- Config: `DQ=128,64,8,5 / DKDV=128,64,8,3 / DKDV_MR=256 / QMR=1984`
- Final head-to-head winner-lock replays:
  - whale `0.293081552297`, FA3 `0.310599344239`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_primary_validate_dq_128_64_8_5_dkdv_128_64_8_3_mr_256_qmr_1984_20260421_014059.json`
  - whale `0.295900591940`, FA3 `0.307649663761`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_primary_validate_dq_128_64_8_5_dkdv_128_64_8_3_mr_256_qmr_1984_20260421_014217.json`

## Strongest displaced challengers

### 2x fallback challenger

- Family: `fallback`
- Config: `DQ=128,64,8,2 / DKDV=128,128,8,3 / DKDV_MR=256 / QMR=928`
- Strongest earlier 2x flash:
  - whale `0.285529936209`, FA3 `0.306176656052`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_fallback_validate_dq_128_64_8_2_dkdv_128_128_8_3_mr_256_qmr_928_20260420_231655.json`
- Final head-to-head replays:
  - whale `0.295171663806`, FA3 `0.300850479931`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_fallback_validate_dq_128_64_8_2_dkdv_128_128_8_3_mr_256_qmr_928_20260421_014059.json`
  - whale `0.296852319911`, FA3 `0.312706207991`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_fallback_validate_dq_128_64_8_2_dkdv_128_128_8_3_mr_256_qmr_928_20260421_014216.json`

### 2x hybrid A insurance lane

- Family: `hybrid_a`
- Config: `DQ=128,64,8,2 / DKDV=128,64,8,3 / DKDV_MR=256 / QMR=2048`
- Best validates observed:
  - whale `0.289068399772`, FA3 `0.299265231654`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_hybrid_a_validate_dq_128_64_8_2_dkdv_128_64_8_3_mr_256_qmr_2048_20260421_001616.json`
  - whale `0.294014768258`, FA3 `0.297577568218`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_hybrid_a_validate_dq_128_64_8_2_dkdv_128_64_8_3_mr_256_qmr_2048_20260421_013507.json`

### 2x hybrid B displaced single-GPU leader

- Family: `hybrid_b`
- Config: `DQ=128,64,8,5 / DKDV=128,128,8,3 / DKDV_MR=256`
- Best 2x validates observed:
  - `QMR=1952`: whale `0.289053536043`, FA3 `0.302906079993`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_hybrid_b_validate_dq_128_64_8_5_dkdv_128_128_8_3_mr_256_qmr_1952_20260420_235947.json`
  - `QMR=2192`: whale `0.289752959877`, FA3 `0.309961471722`
  - evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/evidence/cross_hybrid_b_validate_dq_128_64_8_5_dkdv_128_128_8_3_mr_256_qmr_2192_20260421_000451.json`

## Net result

- Single-GPU carry-forward winner: `hybrid_b @ 1664`
- 2x carry-forward winner: `primary @ 1984`
- Next downstream carry-forward for donor work: `primary @ 1984`
