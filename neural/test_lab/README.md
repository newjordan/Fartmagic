# Test Lab

`test_lab/` is the stable launch surface for runs in `sota_nueral`.

Code ownership:
- Mutable training code lives under `experiments/`
- Frozen records live under `records/`
- `test_lab/` only contains launch wrappers and run-facing notes

Current launch targets:
- `test_lab/Rascal_II_homebase/run.sh`
- `test_lab/Rascal_II_mixed_int_lab/run.sh`
- `test_lab/Rascal_IV/run.sh`
- `test_lab/Rascal_IV/clean.sh`
- `test_lab/Rascal_III_runner2778/run.sh`
- `test_lab/PR_sidelane_8x_linked/run.sh`
- `test_lab/PR_sidelane_8x_linked/run_v1_non_ngram.sh`
- `test_lab/PR_sidelane_8x_linked/run_v1_2_non_ngram.sh`
- `test_lab/PR_sidelane_8x_linked/run_v1_3_non_ngram.sh`
- `test_lab/PR_sidelane_8x_linked/run_longworm.sh`
- `test_lab/PR_sidelane_8x_linked/run_v1_4_power.sh`
- `test_lab/PR_sidelane_8x_linked/run_longworm_submission.sh`
- `test_lab/PR_sidelane_8x_linked/run_longworm_rk.sh`
- `test_lab/today.sh`

Examples:

```bash
bash test_lab/today.sh
```

Today's fixed test:
- Rascal II mixed-int probe
- `SEED=300`
- `QUANT_ATTN_BITS=5`
- `QUANT_MLP_BITS=6`
- `QUANT_AUX_BITS=6`
- `QUANT_EMBED_BITS=8`
- `QUANT_OTHER_BITS=8`
