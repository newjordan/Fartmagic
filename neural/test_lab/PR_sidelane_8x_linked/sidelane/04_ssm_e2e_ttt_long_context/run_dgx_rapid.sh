#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT_DIR}/scripts/run_dgx_spark_rapid_ablation.sh" 04_ssm_e2e_ttt_long_context "${1:-all}"
