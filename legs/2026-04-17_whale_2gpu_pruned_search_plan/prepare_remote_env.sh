#!/usr/bin/env bash
set -euo pipefail

if [[ -x /venv/main/bin/python3 ]]; then
  export PATH="/venv/main/bin:${PATH}"
fi

python3 - <<'PY2'
import sys
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpus={torch.cuda.device_count()}")
if torch.version.cuda != "13.0":
    raise SystemExit(f"Expected CUDA 13.0, got {torch.version.cuda!r}")
PY2

if ! python3 - <<'PY2' >/dev/null 2>&1
import flash_attn_interface
PY2
then
  pip install --no-cache-dir 'https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl'
fi

python3 - <<'PY2'
import flash_attn_interface
print(f"fa3={getattr(flash_attn_interface, '__file__', 'missing')}")
PY2
