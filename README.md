# Fartmagic

Isolated repo for the Midnight neural experiments. Training code lives in frozen vault files and in per-experiment legs. Shared tooling creates those legs, refuses untracked shell overrides, and checks that a trainer edit stays small.

This repository does not include a license file.

## Layout

| Path | Role |
| --- | --- |
| `vault/train_gpt_midnight_12l_sota_REAL.py` | Frozen 12L reference. Do not edit. |
| `vault/train_gpt_midnight_iii_base.py` | Default parent for a new leg. Do not edit. |
| `scripts/Im_sorry_pod_setup.sh` | Frozen pod bootstrap for CUDA 13.0 / PyTorch 2.11.0+cu130. Do not edit. |
| `scripts/pod_stack_guard.sh` | Hash check for that bootstrap. Do not edit. |
| `scripts/new_leg.sh` | Creates `legs/<date>_<name>/`. |
| `scripts/leg_diff_guard.py` | Fails a trainer diff that is wider than the configured limits. |
| `scripts/launch_guard.sh` | Preflight used by leg launchers: stack lock, paths, GPU count, single-run lock. |
| `scripts/agent_guard.sh` | Limits an agent session to an allowlist. |
| `legs/` | One directory per experiment. Each has `train_gpt.py`, `tracked_env.sh`, `gate.sh`, and `run.sh` unless noted below. |
| `data/cached_challenge_fineweb.py` | Downloads FineWeb shards and SentencePiece tokenizers. It does not download model weights. |
| `scripts/run_*` | Competition and architecture launchers that call the trainers in `scripts/`. |

`legs/` currently contains 16 experiment directories. `legs/2026-04-14_midnight_iii_v_bank_gptq_fix` has `run.sh` and no `gate.sh`.

## Promoted artifact

`CURRENT.md` points at `legs/2026-04-13_midnight_iii_v_qattn6`. The tracked log `legs/2026-04-13_midnight_iii_v_qattn6/logs/full_seed444_20260413_044438.log` records:

- line 21: `seed:444`
- line 107: `Total submission size mixed+brotli: 15576180 bytes`
- line 111: `final_sliding_window_exact val_loss:2.81967914 val_bpb:1.09159026`

## Local setup

Python 3.11 or newer. The GPU pod uses Python 3.12.

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

`make test` runs the same suite. The tests do not download datasets or model weights.

Install the dataset helper only when you intend to fetch FineWeb shards:

```bash
python3 -m pip install -e ".[data]"
```

Training imports are not pinned here. A paid run uses the pod image `vastai/pytorch:cuda-13.0.2-auto` with PyTorch 2.11.0+cu130 and the cu130 FlashAttention 3 wheel installed by `scripts/Im_sorry_pod_setup.sh`.

## Pod setup

From the repo root on the pod:

```bash
bash scripts/Im_sorry_pod_setup.sh
bash scripts/pod_stack_guard.sh verify
```

The setup script materializes `data/tokenizers/` and `data/datasets/`. Confirm the repo path and branch before a paid launch:

```bash
pwd
git remote -v
git rev-parse --abbrev-ref HEAD
test -f legs/<leg>/gate.sh
test -f data/tokenizers/fineweb_8192_bpe.model   # or the tokenizer that leg expects
```

Optional pre-commit hook, installed into `.git/hooks/` and not committed:

```bash
bash scripts/install_pod_guard_hook.sh
```

It refuses staged edits to the pod setup script, its lock, and `vault/` unless `ALLOW_POD_STACK_EDIT=1` is set.

## Data

```bash
python3 data/cached_challenge_fineweb.py --help
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
```

Shards land in `data/datasets/`. Tokenizers land in `data/tokenizers/`. Both paths are gitignored. Set `DATA_PATH` and `TOKENIZER_PATH` when the files are not in the default repo-relative locations. Launchers also look in `/dev/shm` and the historical pod checkouts `/root/Fartmagic` and `/workspace/SOTA_FINAL`.

## Create and run a leg

```bash
bash scripts/new_leg.sh qk_gain_probe
# optional parent:
bash scripts/new_leg.sh vocab_trim --source vault/train_gpt_midnight_12l_sota_REAL.py
```

Then edit that leg only:

1. `hypothesis.md` — one variable, and a `Parent:` line.
2. `tracked_env.sh` for an environment change, or `train_gpt.py` for a code change.
3. `python3 scripts/leg_diff_guard.py legs/<leg>`
4. `bash legs/<leg>/gate.sh` for the short gate, or `bash legs/<leg>/run.sh` for the full run.

`gate.sh` and `run.sh` exit if experiment variables such as `NUM_LAYERS` or `DATA_PATH` are already exported in the shell. Change `tracked_env.sh` instead. `SEED` and `NPROC_PER_NODE` are the launch overrides those scripts accept.

Lane wrappers forward those two overrides and then exec the clean campaign:

```bash
bash scripts/run_midnight_iii_lane.sh gate
bash scripts/run_midnight_iii_lane.sh run 444 8
bash scripts/run_midnight_12l_lane.sh gate
```

`gate` and `run` here are the script names. The III wrapper runs `legs/2026-04-12_midnight_iii_clean`. The 12L wrapper runs `legs/2026-04-12_midnight_12l_clean`. The promoted qattn6 leg is started from its own `gate.sh` or `run.sh`.

## Competition runners

These scripts live in `scripts/` and train `scripts/train_gpt.py` or `scripts/train_gpt_rwkv_moonshot.py`. They are separate from the leg workflow.

```bash
DRY_RUN=1 bash scripts/run_competition.sh
DRY_RUN=1 bash scripts/run_competition_moonshot.sh
bash scripts/run_moonshot_8k.sh
bash scripts/run_nc3_8k_stacked_4loop_anchor.sh
```

`DRY_RUN=1` prints the resolved profile and does not start training, including when the dataset is absent. A real launch still requires `DATA_PATH` and `TOKENIZER_PATH` to exist. `WORLD_SIZE` greater than 1 uses `python3 -m torch.distributed.run`.

## Diff guard

```bash
python3 scripts/leg_diff_guard.py legs/<leg>
python3 scripts/leg_diff_guard.py legs/<leg> --parent vault/train_gpt_midnight_iii_base.py --no-diff
python3 scripts/leg_diff_guard.py legs/<leg> --write-lock
python3 scripts/leg_diff_guard.py legs/<leg> --check-lock
```

Defaults: at most one changed hyperparameter environment variable, no other hyperparameter edits, no code edits outside `class Hyperparameters`, and at most 40 changed non-blank lines. A leg whose `gate.sh` passes wider limits is declaring that wider footprint on purpose. `--check-lock` compares both the trainer hash and the parent hash stored in the lock.

Environment reads counted by the guard are `os.environ.get`, `os.environ.pop`, and `os.environ["NAME"]`, with single or double quotes.

## Agent guard

```bash
bash scripts/agent_guard.sh status
bash scripts/agent_guard.sh snapshot pre_change
bash scripts/agent_guard.sh check legs/<leg>/agent_allowlist.txt --delta-latest
```

Allowlist lines are exact paths, prefixes ending in `/`, or shell globs. Quoted git paths are unquoted before the check.

## Tests and CI

GitHub Actions runs `python -m pytest` on Python 3.12 for pushes and pull requests (`.github/workflows/ci.yml`). The suite covers the diff guard, FineWeb path rules, leg creation, launch-guard GPU and launcher checks, the agent allowlist, the pod stack hash, and shell syntax of the scripts under `scripts/` and `legs/`.

## Frozen files

Do not edit these without an explicit decision to change the pod stack or the reference trainers:

- `scripts/Im_sorry_pod_setup.sh`
- `scripts/Im_sorry_pod_setup.sh.cu130_frozen_20260414`
- `scripts/pod_stack_guard.sh`
- `scripts/pod_stack.lock`
- `vault/`
