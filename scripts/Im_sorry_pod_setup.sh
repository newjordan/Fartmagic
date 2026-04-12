#!/bin/bash
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore   # suppress "running as root" pip warning
# =============================================================================
# POD SETUP — the only script you ever run on a pod
#
# Usage:  bash scripts/Im_sorry_pod_setup.sh
#   (or curl from raw URL and pipe to bash — works either way)
#
# What it does:
#   1. Clones/syncs repo to the 'test' branch
#   2. Installs deps (pip, zstandard, FA3, dataset)
#   3. Verifies everything works
#   4. Done. You run your experiment manually.
# =============================================================================

REPO_URL="https://github.com/newjordan/parameter-golf.git"
BRANCH="${BRANCH:-TEST_LAB}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
DATASET_VARIANT="${DATASET_VARIANT:-sp1024}"
ACTIVATE_HELPER_REL="scripts/activate_pod_env.sh"
if [[ -x /venv/main/bin/python3 ]]; then
    export PATH="/venv/main/bin:${PATH}"
fi
# Auto-detect repo root from script location; fall back for curl-pipe scenario
_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)" || true
_CANDIDATE="$(cd -- "${_SCRIPT_DIR}/.." && pwd 2>/dev/null)" || true
if [[ -d "${_CANDIDATE}/.git" ]]; then
    WORKSPACE="${_CANDIDATE}"
else
    WORKSPACE="/workspace/parameter-golf"
fi

echo "============================================"
echo "  POD SETUP"
echo "  Branch: ${BRANCH}"
echo "  Variant: ${DATASET_VARIANT}"
echo "  Train shards: ${TRAIN_SHARDS}"
echo "============================================"

# =============================================================================
# 1. Get the repo on the test branch
# =============================================================================
if [ -d "${WORKSPACE}/.git" ]; then
    echo "[1/6] Repo exists, force-syncing to ${BRANCH}..."
    cd "${WORKSPACE}"
    git fetch origin "${BRANCH}" --quiet
    git checkout -B "${BRANCH}" "origin/${BRANCH}" --force
    git clean -fd --quiet
elif [ -d "${WORKSPACE}" ]; then
    echo "[1/6] Existing non-git workspace detected, using in-place files..."
    cd "${WORKSPACE}"
else
    echo "[1/6] Cloning repo..."
    git clone -b "${BRANCH}" "${REPO_URL}" "${WORKSPACE}"
    cd "${WORKSPACE}"
fi
if [ -d "${WORKSPACE}/.git" ]; then
    echo "  HEAD: $(git log --oneline -1)"
else
    echo "  HEAD: non-git workspace (no commit metadata)"
fi

# =============================================================================
# 2. Verify base environment (system Python + PyTorch must already exist)
# =============================================================================
echo ""
echo "[2/6] Checking base environment..."

python3 --version || { echo "FATAL: python3 not found"; exit 1; }
python3 - <<'PYEOF'
import sys
import torch

torch_version = torch.__version__
cuda_version = str(torch.version.cuda or "")
print(f"  PyTorch {torch_version}  CUDA {cuda_version}")
if "+cu124" in torch_version or cuda_version.startswith("12.4"):
    raise SystemExit("FATAL: stale cu124 torch stack detected; use the correct pod image before running this setup.")
PYEOF
TORCH_LIB="$(python3 - <<'PYEOF'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYEOF
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
ACTIVATE_HELPER="${WORKSPACE}/${ACTIVATE_HELPER_REL}"
cat > "${ACTIVATE_HELPER}" <<'ACTEOF'
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
TORCH_LIB="$(python3 - <<'PYEOF'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYEOF
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
if [[ -d "${REPO_ROOT}/flash-attention/hopper" ]]; then
    export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
fi
ACTEOF
chmod +x "${ACTIVATE_HELPER}"

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "  WARNING: No GPUs detected"
else
    python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_mem // 1024**3}GB)')
" 2>/dev/null || true
fi

# =============================================================================
# 3. Core pip packages (system site-packages, no conda, no PYTHONPATH)
# =============================================================================
echo ""
echo "[3/6] Installing pip packages..."

pip install --upgrade pip -q 2>&1 | tail -1

pip install numpy tqdm huggingface-hub kernels setuptools \
    "typing-extensions==4.15.0" datasets tiktoken sentencepiece attr -q 2>&1 | tail -1
echo "  Core packages OK"

# =============================================================================
# 4. zstandard (CRITICAL: prevents artifact size inflation)
# =============================================================================
echo ""
echo "[4/6] zstandard..."

if python3 -c "import zstandard" 2>/dev/null; then
    echo "  Already installed"
else
    pip install zstandard -q
    echo "  Installed"
fi
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__}')"

echo "  brotli..."
if python3 -c "import brotli" 2>/dev/null; then
    echo "  Already installed"
else
    pip install brotli -q
    echo "  Installed"
fi
python3 -c "import brotli; print(f'  brotli {brotli.__version__}')" 2>/dev/null || echo "  brotli OK"

# =============================================================================
# 5. FlashAttention-3
# =============================================================================
echo ""
echo "[5/6] FlashAttention-3..."

sync_fa3_dir_into_site() {
    local fa_dir="$1"
    local site_dir
    local linked=0
    site_dir="$(python3 -c "import site; print(site.getsitepackages()[0])")"

    for pattern in flash_attn_interface.py flash_attn_interface*.so flash_attn_config.py; do
        for src in "${fa_dir}"/${pattern}; do
            [ -e "${src}" ] || continue
            ln -sf "${src}" "${site_dir}/$(basename "${src}")"
            linked=1
        done
    done

    if [ "${linked}" -eq 1 ]; then
        echo "  Symlinked FA3 runtime from ${fa_dir} into ${site_dir}"
        return 0
    fi
    return 1
}

find_system_fa3_dir() {
    for py in $(which -a python3 2>/dev/null | awk '!seen[$0]++') /opt/conda/bin/python3 /usr/bin/python3; do
        [ -x "${py}" ] || continue
        fa_dir="$("${py}" - <<'PYEOF' 2>/dev/null || true
import inspect
import os
try:
    import flash_attn_interface
    print(os.path.dirname(inspect.getfile(flash_attn_interface)))
except Exception:
    pass
PYEOF
)"
        if [ -n "${fa_dir}" ]; then
            printf '%s\n' "${fa_dir}"
            return 0
        fi
    done
    return 1
}

install_fa3() {
    echo "  Searching system for pre-installed flash_attn_interface..."
    local system_fa3_dir=""
    system_fa3_dir="$(find_system_fa3_dir || true)"
    if [ -n "${system_fa3_dir}" ] && sync_fa3_dir_into_site "${system_fa3_dir}"; then
        return 0
    fi

    echo "  Attempting FA3 abi3 wheel (cu128)..."
    if pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        2>&1 | tail -3; then
        return 0
    fi

    echo "  Wheels failed. Checking for local flash-attention/hopper source..."
    if [ -d "${WORKSPACE}/flash-attention/hopper" ]; then
        if sync_fa3_dir_into_site "${WORKSPACE}/flash-attention/hopper"; then
            return 0
        fi
    fi

    echo "  FATAL: Could not install or locate FA3 for the current torch stack."
    return 1
}

if python3 -c "from flash_attn_interface import flash_attn_func; print('  FA3 (flash_attn_interface) OK')" 2>/dev/null; then
    : # already good
elif python3 -c "import flash_attn; v=flash_attn.__version__; assert v.startswith('3'); print(f'  FA3 v{v} OK')" 2>/dev/null; then
    : # flash_attn v3 package works
else
    install_fa3
fi
python3 - <<'PYEOF'
import importlib
importlib.import_module("flash_attn_3._C")
from flash_attn_interface import flash_attn_func  # noqa: F401
print("  FA3 runtime OK")
PYEOF

# =============================================================================
# 6. Dataset (sp1024)
# =============================================================================
echo ""
echo "[6/6] Tokenizer + FineWeb dataset (${DATASET_VARIANT})..."

# SP1024: official competition repo (willdepueoai/parameter-golf)
echo "  Downloading SP1024 (competition default)..."
cd "${WORKSPACE}"
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
echo "  SP1024 data downloaded"

# SP8192: from kevclark/parameter-golf (merged leaderboard submission)
echo "  Downloading SP8192 (kevclark/parameter-golf)..."
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards "${TRAIN_SHARDS}" || {
    echo "  WARNING: SP8192 download failed (disk space or network). SP1024 still available."
}
# Restore default manifest for future sp1024 downloads
rm -f data/manifest.json
echo "  Dataset setup complete"

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "============================================"
echo " Verification"
echo "============================================"

python3 - << 'PYEOF'
import os, sys, glob

print(f"Python       : {sys.version.split()[0]}")
print(f"Executable   : {sys.executable}")

import torch
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

fa = "NOT FOUND"
try:
    from flash_attn_interface import flash_attn_func
    fa = "flash_attn_interface (FA3 hopper)"
except ImportError:
    try:
        import flash_attn
        v = flash_attn.__version__
        fa = f"flash_attn v{v}" + ("" if v.startswith("3") else " WARNING: not FA3!")
    except ImportError:
        pass
print(f"FlashAttn    : {fa}")

try:
    import zstandard
    print(f"zstandard    : {zstandard.__version__}")
except ImportError:
    print("zstandard    : MISSING!")

try:
    import sentencepiece
    print(f"sentencepiece: OK")
except ImportError:
    print("sentencepiece: MISSING!")

variant = os.environ.get("DATASET_VARIANT", "sp1024")
dataset_dir = "fineweb10B_byte260" if variant == "byte260" else f"fineweb10B_{variant}"
train = sorted(glob.glob(f"./data/datasets/{dataset_dir}/fineweb_train_*.bin"))
val   = sorted(glob.glob(f"./data/datasets/{dataset_dir}/fineweb_val_*.bin"))
print(f"Train shards : {len(train)}")
print(f"Val shards   : {len(val)}")
PYEOF

echo ""
echo "============================================"
echo " READY."
echo "============================================"
echo "Activation helper: ${ACTIVATE_HELPER_REL}"
