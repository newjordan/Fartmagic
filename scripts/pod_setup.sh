#!/bin/bash
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore   # suppress "running as root" pip warning
# =============================================================================
# POD SETUP — the only script you ever run on a pod
#
# Usage:  bash pod_setup.sh
#   (or curl from raw URL and pipe to bash — works either way)
#
# What it does:
#   1. Clones/syncs repo to the 'test' branch
#   2. Installs deps (pip, zstandard, FA3, dataset)
#   3. Verifies everything works
#   4. Done. You run your experiment manually.
# =============================================================================

REPO_URL="https://github.com/newjordan/parameter-golf.git"
BRANCH="TEST_LAB"
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
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA {torch.version.cuda}')" \
    || { echo "FATAL: PyTorch not installed in system Python"; exit 1; }

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

# =============================================================================
# 5. FlashAttention-3
# =============================================================================
echo ""
echo "[5/6] FlashAttention-3..."

fa3_runtime_check() {
    python3 - << 'PYEOF' >/dev/null 2>&1
import importlib
from flash_attn_interface import flash_attn_func  # noqa: F401
importlib.import_module("flash_attn_3._C")
PYEOF
}

install_fa3() {
    # --- 1. Official FA3 abi3 wheel for cu124 ---
    _fa3_whl_url="https://download.pytorch.org/whl/cu124/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
    echo "  Trying official FA3 abi3 wheel (cu124)..."
    if pip install --no-cache-dir "${_fa3_whl_url}" 2>&1 | tail -3; then
        if fa3_runtime_check; then
            echo "  Installed and verified FA3 wheel"
            return 0
        fi
        echo "  Wheel installed but runtime ABI check failed; continuing..."
    fi

    # --- 2. Search system for pre-installed FA3 runtime and bridge it ---
    echo "  Searching system for FA3 runtime (_C.abi3.so + flash_attn_interface.py)..."
    _fa3_path=""
    for _py in $(which -a python3 2>/dev/null | awk '!seen[$0]++') /opt/conda/bin/python3 /usr/bin/python3; do
        [ -x "${_py}" ] || continue
        _fa3_path=$("${_py}" -c "
import inspect, os
try:
    import flash_attn_interface
    print(os.path.dirname(inspect.getfile(flash_attn_interface)))
except ImportError:
    pass
" 2>/dev/null)
        if [ -n "${_fa3_path}" ]; then
            SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
            echo "  Found FA3 at ${_fa3_path} (via ${_py})"
            mkdir -p "${SITE}/flash_attn_3"
            for _f in "${_fa3_path}"/flash_attn_interface*; do
                [ -e "${_f}" ] && ln -sf "${_f}" "${SITE}/"
            done
            for _cand in "${_fa3_path}"/flash_attn_3/_C*.so \
                         /usr/local/lib/python3.*/dist-packages/flash_attn_3/_C*.so \
                         /opt/conda/lib/python3.*/site-packages/flash_attn_3/_C*.so; do
                [ -e "${_cand}" ] || continue
                ln -sf "${_cand}" "${SITE}/flash_attn_3/_C.abi3.so"
                touch "${SITE}/flash_attn_3/__init__.py"
                break
            done
            if fa3_runtime_check; then
                echo "  Bridged and verified FA3 runtime from system packages"
                return 0
            fi
            echo "  Found candidate FA3, but runtime ABI check failed; continuing..."
        fi
    done

    # --- 3. Local flash-attention/hopper source ---
    echo "  Checking for local flash-attention/hopper source..."
    if [ -d "${WORKSPACE}/flash-attention/hopper" ]; then
        SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
        SRC="${WORKSPACE}/flash-attention/hopper/flash_attn_interface.py"
        if [ -f "$SRC" ]; then
            ln -sf "$SRC" "${SITE}/flash_attn_interface.py"
            if fa3_runtime_check; then
                echo "  Symlinked and verified flash_attn_interface.py runtime"
                return 0
            fi
            echo "  Local flash_attn_interface exists, but runtime ABI check failed; continuing..."
        fi
    fi

    echo "  FATAL: Could not install a working FA3 runtime."
    return 1
}

if fa3_runtime_check; then
    echo "  FA3 runtime already valid"
elif python3 -c "import flash_attn; v=flash_attn.__version__; assert v.startswith('3'); print(f'  FA3 v{v} OK')" 2>/dev/null && fa3_runtime_check; then
    : # already good
else
    install_fa3 || exit 1
fi

if ! fa3_runtime_check; then
    echo "FATAL: FA3 is required but not runtime-valid (flash_attn_3._C import failed)."
    exit 1
fi

# =============================================================================
# 6. Dataset (sp1024)
# =============================================================================
echo ""
echo "[6/6] Tokenizer + FineWeb dataset (sp1024)..."

# Tokenizer
TOKENIZER="${WORKSPACE}/data/tokenizers/fineweb_1024_bpe.model"
if [ -f "${TOKENIZER}" ]; then
    echo "  Tokenizer already present"
else
    echo "  Downloading tokenizer..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download sproos/parameter-golf-tokenizers \
            --include "tokenizers/*" --local-dir "${WORKSPACE}/data"
    else
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sproos/parameter-golf-tokenizers',
    allow_patterns='tokenizers/*',
    local_dir='${WORKSPACE}/data')
"
    fi
    echo "  Tokenizer downloaded"
fi

# Dataset shards — use nullglob array so unmatched glob = 0, not a crash
shopt -s nullglob
_train=("${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_train_"*.bin)
_val=("${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_val_"*.bin)
TRAIN_COUNT=${#_train[@]}
VAL_COUNT=${#_val[@]}
shopt -u nullglob

if [ "$TRAIN_COUNT" -ge 10 ]; then
    echo "  Already have $TRAIN_COUNT train / $VAL_COUNT val shards"
else
    echo "  Downloading dataset ($TRAIN_COUNT train shards found, need 10+)..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download sproos/parameter-golf-tokenizers \
            --include "datasets/fineweb10B_sp1024/*" --local-dir "${WORKSPACE}/data"
    else
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sproos/parameter-golf-tokenizers',
    allow_patterns='datasets/fineweb10B_sp1024/*',
    local_dir='${WORKSPACE}/data')
"
    fi
    echo "  Dataset downloaded"
fi

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "============================================"
echo " Verification"
echo "============================================"

python3 - << 'PYEOF'
import sys, glob

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

train = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))
val   = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
print(f"Train shards : {len(train)}")
print(f"Val shards   : {len(val)}")
PYEOF

echo ""
echo "============================================"
echo " READY."
echo "============================================"
