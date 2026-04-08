#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

exec bash crawler/2026-04-02_BW19_CrawlerSystem_2k/run_ablation_sequence.sh
