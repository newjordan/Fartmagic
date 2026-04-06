#!/usr/bin/env bash
set -euo pipefail
RASCAL_IV_ARM=control exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run.sh" "$@"
