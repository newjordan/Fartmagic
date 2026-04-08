#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

python3 "${REPO_ROOT}/experiments/slot_micro_lab/slot_micro_harness.py" --steps 10 --seed 1337
