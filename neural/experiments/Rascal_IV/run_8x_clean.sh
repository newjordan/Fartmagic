#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Clean/stable Rascal IV profile:
# - control arm only
# - no GPTQ calibration pass
# - no n-gram eval passes
# - skip quant roundtrip eval (export artifact still generated)
# - keep export reserve so wallclock training behavior stays consistent
export RASCAL_IV_ARM="control"
export RASCAL_IV_ALLOW_EXPERIMENTAL="0"
export SKIP_GPTQ="${SKIP_GPTQ:-1}"
export EXPORT_RESERVE_MS="${EXPORT_RESERVE_MS:-0}"
export NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-0}"
export NGRAM_EVAL_MAX_SECONDS="${NGRAM_EVAL_MAX_SECONDS:-0}"
export NGRAM_ENTROPY_SHIFT="${NGRAM_ENTROPY_SHIFT:-0}"
export QUANT_ROUNDTRIP_EVAL="${QUANT_ROUNDTRIP_EVAL:-0}"

exec bash "${SCRIPT_DIR}/run_8x.sh" "$@"
