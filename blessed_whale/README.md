# Blessed Whale

This folder is the clean carry-forward bundle for the blessed whale donor+kernel path.

It replaces the older scattered `wh_kern_int` control plane with one self-contained bundle named for the thing we actually want to run.

## Carry-forward default
- donor: `whale_l12_dim576_matrixbits5`
- trainer contract: `pr1493_decompressed.py`
- attention mode: `whale_native`
- blessed kernel family: `primary`
- `DQ=128,64,8,5`
- `DKDV=128,64,8,3`
- `DKDV_MR=256`
- `QMR=1984`

## Benchmark ladder
- `B0`: same-contract FA3 baseline
- `B1`: blessed whale native default
- `B2`: fallback kernel recovery
- `B3`: hybrid diagnostic only

## Canonical entrypoints
- runner: `blessed_whale/blessed_whale_train_gpt.py`
- env: `blessed_whale/blessed_whale_env.sh`
- launcher: `blessed_whale/run_blessed_whale.sh`
- matrix: `blessed_whale/blessed_whale_benchmark_matrix.json`
- blessed anchor: `blessed_whale/blessed_whale_anchor.json`
- fallback anchor: `blessed_whale/blessed_whale_fallback_anchor.json`

## Usage
```bash
BLESSED_WHALE_BENCHMARK_ID=B0 bash blessed_whale/run_blessed_whale.sh
BLESSED_WHALE_BENCHMARK_ID=B1 bash blessed_whale/run_blessed_whale.sh
BLESSED_WHALE_BENCHMARK_ID=B2 bash blessed_whale/run_blessed_whale.sh
BLESSED_WHALE_BENCHMARK_ID=B3 bash blessed_whale/run_blessed_whale.sh
```

## Non-negotiable rules
- use `pr1493_decompressed.py`
- do not revive `gemini_runs/pr1493_train_gpt.py`
- do not switch to `scripts/train_gpt.py`
- do not hand-edit whale kernel env vars in the shell to select profiles
- use the benchmark ladder instead of ad hoc recovery
