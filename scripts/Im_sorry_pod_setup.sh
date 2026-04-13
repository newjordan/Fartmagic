#!/usr/bin/env bash
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore

# =============================================================================
# POD SETUP — focused bootstrap for a fresh pod
#
# Usage:
#   cd /path/to/project
#   bash scripts/pod_setup.sh
#
# What it does:
#   1. Uses the pod's working Python/Torch stack as-is
#   2. Installs core Python deps plus zstandard/brotli/python-minifier
#   3. Installs or wires FlashAttention-3 if available
#   4. Downloads SP8192 (large vocab) and SP1024 data if the local data helper exists
#   5. Verifies the environment and both vocab setups
#
# What it deliberately does NOT do:
#   - no git pull
#   - no branch checkout
#   - no repo clean
#   - no cloning
# =============================================================================

TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
DOWNLOAD_SP1024="${DOWNLOAD_SP1024:-1}"
DOWNLOAD_SP8192="${DOWNLOAD_SP8192:-1}"
export TRAIN_SHARDS DOWNLOAD_SP1024 DOWNLOAD_SP8192

if [[ -x /venv/main/bin/python3 ]]; then
    export PATH="/venv/main/bin:${PATH}"
fi

if [[ -n "${WORKSPACE:-}" ]]; then
    WORKSPACE="$(cd -- "${WORKSPACE}" && pwd)"
else
    WORKSPACE="$(pwd)"
fi
export WORKSPACE

DATA_HELPER="${WORKSPACE}/data/cached_challenge_fineweb.py"
FA3_SRC="${WORKSPACE}/flash-attention/hopper/flash_attn_interface.py"

echo "============================================"
echo "  POD SETUP"
echo "  Workspace: ${WORKSPACE}"
echo "  Train shards: ${TRAIN_SHARDS}"
echo "  Repo sync: DISABLED"
echo "============================================"

# =============================================================================
# 1. Verify base environment
# =============================================================================
echo ""
echo "[1/5] Checking base environment..."

python3 --version || { echo "FATAL: python3 not found"; exit 1; }
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA {torch.version.cuda}')" \
    || { echo "FATAL: PyTorch not installed in active python3"; exit 1; }

TORCH_LIB="$(python3 - <<'PYEOF'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYEOF
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

GPU_COUNT="$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")"
if [[ "${GPU_COUNT}" -eq 0 ]]; then
    echo "  WARNING: No GPUs detected"
else
    python3 - <<'PYEOF' 2>/dev/null || true
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory // 1024**3}GB)")
PYEOF
fi

if [[ -d /workspace/venv_cu124 ]]; then
    echo "  NOTE: ignoring stale /workspace/venv_cu124; setup uses the pod's active python3"
fi

# =============================================================================
# 2. Core pip packages
# =============================================================================
echo ""
echo "[2/5] Installing pip packages..."

pip install --upgrade pip -q 2>&1 | tail -1

pip install numpy tqdm huggingface-hub kernels setuptools \
    "typing-extensions==4.15.0" datasets tiktoken sentencepiece attr python-minifier -q 2>&1 | tail -1
echo "  Core packages OK"

# =============================================================================
# 3. Compression + tokenizer libs
# =============================================================================
echo ""
echo "[3/5] Compression + tokenizer libraries..."

if python3 -c "import zstandard" 2>/dev/null; then
    echo "  zstandard: already installed"
else
    pip install zstandard -q
    echo "  zstandard: installed"
fi
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__}')"

if python3 -c "import brotli" 2>/dev/null; then
    echo "  brotli: already installed"
else
    pip install brotli -q
    echo "  brotli: installed"
fi
python3 -c "import brotli; print(f'  brotli {brotli.__version__}')" 2>/dev/null || echo "  brotli OK"

if python3 -c "import sentencepiece" 2>/dev/null; then
    echo "  sentencepiece: already installed"
else
    pip install sentencepiece -q
    echo "  sentencepiece: installed"
fi
python3 -c "import sentencepiece; print('  sentencepiece OK')"

if python3 -c "import python_minifier" 2>/dev/null; then
    echo "  python-minifier: already installed"
else
    pip install python-minifier -q
    echo "  python-minifier: installed"
fi
python3 -c "import python_minifier; print('  python-minifier OK')"

# =============================================================================
# 4. FlashAttention-3
# =============================================================================
echo ""
echo "[4/5] FlashAttention-3..."

install_fa3() {
    echo "  Attempting FA3 abi3 wheel (cu128)..."
    if pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        2>&1 | tail -3; then
        return 0
    fi

    echo "  Wheels failed. Checking for local flash-attention/hopper source..."
    if [[ -f "${FA3_SRC}" ]]; then
        SITE="$(python3 -c "import site; print(site.getsitepackages()[0])")"
        ln -sf "${FA3_SRC}" "${SITE}/flash_attn_interface.py"
        echo "  Symlinked flash_attn_interface.py into site-packages"
        return 0
    fi

    echo "  WARNING: Could not install FA3. Will fall back to PyTorch SDPA."
    return 1
}

if python3 -c "from flash_attn_interface import flash_attn_func; print('  FA3 (flash_attn_interface) OK')" 2>/dev/null; then
    :
elif python3 -c "import flash_attn; v=flash_attn.__version__; assert v.startswith('3'); print(f'  FA3 v{v} OK')" 2>/dev/null; then
    :
else
    install_fa3
fi

# =============================================================================
# 5. Tokenizers + datasets
# =============================================================================
echo ""
echo "[5/5] Tokenizers + FineWeb datasets..."

if [[ ! -f "${DATA_HELPER}" ]]; then
    echo "  WARNING: data helper not found at ${DATA_HELPER}"
    echo "  Skipping dataset download. Run this script from a project root that has data/cached_challenge_fineweb.py"
else
    cd "${WORKSPACE}"

    if [[ "${DOWNLOAD_SP1024}" == "1" ]]; then
        echo "  Downloading SP1024..."
        python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}" || {
            echo "  WARNING: SP1024 download failed"
        }
    else
        echo "  Skipping SP1024 download (DOWNLOAD_SP1024=${DOWNLOAD_SP1024})"
    fi

    if [[ "${DOWNLOAD_SP8192}" == "1" ]]; then
        echo "  Downloading SP8192..."
        rm -f data/manifest.json
        MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
        python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards "${TRAIN_SHARDS}" || {
            echo "  WARNING: SP8192 download failed"
        }
        rm -f data/manifest.json
    else
        echo "  Skipping SP8192 download (DOWNLOAD_SP8192=${DOWNLOAD_SP8192})"
    fi
fi

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "============================================"
echo " Verification"
echo "============================================"

python3 - <<'PYEOF'
import glob
import os
import sys

root = os.environ.get("WORKSPACE", os.getcwd())

print(f"Python       : {sys.version.split()[0]}")
print(f"Executable   : {sys.executable}")
print(f"Workspace    : {root}")

import torch
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

fa = "NOT FOUND"
try:
    from flash_attn_interface import flash_attn_func  # noqa: F401
    fa = "flash_attn_interface (FA3 hopper)"
except ImportError:
    try:
        import flash_attn
        v = flash_attn.__version__
        fa = f"flash_attn v{v}" + ("" if v.startswith("3") else " WARNING: not FA3!")
    except ImportError:
        pass
print(f"FlashAttn    : {fa}")

for module_name in ("zstandard", "brotli", "sentencepiece", "python_minifier"):
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "OK")
        print(f"{module_name:<13}: {version}")
    except ImportError:
        print(f"{module_name:<13}: MISSING")

def check_tokenizer(filename: str, expected_vocab: int) -> bool:
    path = os.path.join(root, "data", "tokenizers", filename)
    label = f"tokenizer:{filename}"
    if not os.path.isfile(path):
        print(f"{label}: MISSING")
        return False
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=path)
        actual = int(sp.vocab_size())
        status = "OK" if actual == expected_vocab else f"BAD_VOCAB({actual})"
        print(f"{label}: {status}")
        return status == "OK"
    except Exception as exc:
        print(f"{label}: ERROR ({exc})")
        return False

def check_dataset(variant: str) -> bool:
    dataset_dir = os.path.join(root, "data", "datasets", f"fineweb10B_{variant}")
    train = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_train_*.bin")))
    val = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_*.bin")))
    status = "OK" if train and val else "MISSING_OR_INCOMPLETE"
    print(f"dataset:{variant}: train={len(train)} val={len(val)} status={status}")
    return status == "OK"

check_tokenizer("fineweb_1024_bpe.model", 1024)
sp8192_tok_ok = check_tokenizer("fineweb_8192_bpe.model", 8192)
check_dataset("sp1024")
sp8192_data_ok = check_dataset("sp8192")

if os.environ.get("DOWNLOAD_SP8192", "1") == "1" and not (sp8192_tok_ok and sp8192_data_ok):
    raise SystemExit("FATAL: SP8192 is required but missing or incomplete.")
PYEOF

echo ""
echo "============================================"
echo " READY."
echo "============================================"
