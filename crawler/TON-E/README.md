# TON-E: Our Crawler Runner With PR-1579 Rhythm

This folder uses our stable Nightcrawler Cubed runner as the base and overlays a TON-E rhythm preset.

## Goal

- Keep our proven training/eval/quant harness.
- Apply the PR-1579-style layer rhythm in our script, not his script.
- Track concept notes separately in `NOTES_pr1579_concepts.md`.

## What Changed In `train_gpt.py`

- Base remains our runner (`records/track_10min_16mb/2026-04-10_Nightcrawler_Cubed_8xH100/train_gpt.py`).
- Added TON-E rhythm profile (`TON_E_RHYTHM=1` by default in this folder):
  - defaults to `3F + 2C x2` if `NUM_*` values are not explicitly set
  - defaults to `XSA_INCLUDE_FLAT=1` so `XSA_LAST_N` can span flat+crawler blocks
- Baseline can be restored instantly with `TON_E_RHYTHM=0`.

## TON-E Rhythm Knobs

- `TON_E_RHYTHM` (default `1`)
- `TON_E_NUM_FLAT_LAYERS` (default `3`)
- `TON_E_NUM_CRAWLER_LAYERS` (default `2`)
- `TON_E_CRAWLER_LOOPS` (default `2`)
- `TON_E_XSA_LAST_N` (default `F+C`)
- `TON_E_XSA_INCLUDE_FLAT` (default `1`)
- `TON_E_CRAWLER_LOOP_ROPE_SCALES` (optional override)

Explicit runner env vars still win:
- `USE_CRAWLER`, `NUM_FLAT_LAYERS`, `NUM_CRAWLER_LAYERS`, `CRAWLER_LOOPS`, `XSA_LAST_N`, `XSA_INCLUDE_FLAT`

## Example: TON-E Rhythm On Our Runner

```bash
cd crawler/TON-E
bash run_competition.sh
```

## Competition Flow

```bash
cd crawler/TON-E
# 1) Single-seed legal run (defaults: seed=4, 600s, loop-aware GPTQ, prune to 16MB)
bash run_competition.sh

# 2) Three-seed batch and auto-generate submission.json
bash run_competition_3seeds.sh

# 3) Rebuild submission.json from any matching logs
python3 build_submission_json.py --log-glob "logs/tone_comp_s*.txt" --output submission.json
```

## Multi-GPU Launch (Competition)

```bash
cd crawler/TON-E
WORLD_SIZE=8 \
SUBMISSION_HARDWARE="8xH100 SXM" \
bash run_competition_3seeds.sh
```

## Dry Run / Sanity

```bash
cd crawler/TON-E
DRY_RUN=1 bash run_competition.sh
```

## Optional: Revert To Nightcrawler Cubed Shape

```bash
cd crawler/TON-E
TON_E_RHYTHM=0 \
NUM_FLAT_LAYERS=7 \
NUM_CRAWLER_LAYERS=3 \
CRAWLER_LOOPS=3 \
bash run_competition.sh
```
