# PR Sidelane Linked Tests

Launchers:

- Full linked sweep (all categories): `run.sh`
- V1 submission-ready single-lane (non-ngram): `run_v1_non_ngram.sh`
- V1.2 submission-ready single-lane (non-ngram + brotli + tuned MLP/quant): `run_v1_2_non_ngram.sh`
- V1.3 submission sweep (non-ngram + brotli + depth/schedule arms): `run_v1_3_non_ngram.sh`
- Longworm V1.4 power sweep (higher dim/layers + higher matrix lr): `run_longworm.sh` (`run_v1_4_power.sh` alias)
- Longworm submission-safe launcher (default arm: `35_v1_5_longworm_context_mech_l11_d528_h12_kv4_non_ngram_brotli` with FA3-safe `480d` geometry, no forced 4K metric): `run_longworm_submission.sh`
- Longworm fast L8 submission launcher (RK constrained + longctx taps): `run_longworm_fast_l8_submission.sh`
- Longworm local single-GPU SSM+TTT candidate launcher: `run_longworm_local_ssm_ttt.sh`
- Longworm v1.9 adaptive ablation launcher (baseline vs loss-gated TTT vs loss+drift-gated TTT): `run_longworm_v1_9_adaptive.sh`
- Longworm RK concepts sweep (k2 / k4 / hybrid, dedicated Longworm trainer): `run_longworm_rk.sh`

V1 writeup:

- `V1_NON_NGRAM_WRITEUP.md`
- `V1_2_NON_NGRAM_BROTLI_WRITEUP.md`
- `V1_3_NON_NGRAM_BROTLI_SWEEP_WRITEUP.md`
- `LONGWORM_V1_4_POWER_SWEEP_WRITEUP.md`
- `LONGWORM_V1_5_CONTEXT_MECH_WRITEUP.md`
- `LONGWORM_RK_CONCEPTS_WRITEUP.md`
- `LONGWORM_V1_9_ADAPTIVE_TTT_WRITEUP.md`
