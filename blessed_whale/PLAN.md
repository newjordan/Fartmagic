# PLAN — Blessed Whale

## Goal
Run the best donor with the blessed whale kernel through a clean, named, self-contained bundle.

## Carry-forward choice
- donor: `whale_l12_dim576_matrixbits5`
- contract: `pr1493_decompressed.py`
- blessed 2x winner: `primary @ 1984`

## Executable ladder
- `B0`: FA3 baseline on the same donor and trainer contract
- `B1`: blessed whale native path
- `B2`: fallback kernel recovery
- `B3`: hybrid diagnostic only

## Immediate sequence
1. Run `B0` to establish the preserved same-contract donor baseline.
2. Run `B1` as the real blessed whale path.
3. If `B1` fails or regresses badly, run `B2`.
4. Only if `B2` also fails, run `B3` to diagnose native-whale vs donor integration.

## Success condition
- the bundle runs without touching the banned runner
- benchmark selection is explicit and reproducible
- artifacts export under `artifacts/blessed_whale/`
