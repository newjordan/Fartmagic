#!/usr/bin/env bash
set -euo pipefail
RASCAL_IV_ARM=e2e_ttt RASCAL_IV_ALLOW_EXPERIMENTAL=1 exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run.sh" "$@"
