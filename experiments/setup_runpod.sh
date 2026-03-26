#!/bin/bash
# -------------------------------------------------------------------------------
# Parameter Golf -- B-Wing Pod Setup (sp1024 + FA3 + zstandard)
# Run: bash experiments/setup_runpod.sh
# -------------------------------------------------------------------------------

set -e

echo "----------------------------------------------"
echo " Parameter Golf -- B-Wing Pod Setup"
echo "----------------------------------------------"

# -------------------------------------------------------------------------------
# 1. Miniconda
# -------------------------------------------------------------------------------
echo ""
echo "[1/6] Miniconda..."

if [ -d "$HOME/miniconda3" ]; then
    echo "    Already installed -- skipping."
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b
    rm /tmp/miniconda.sh
    ~/miniconda3/bin/conda init bash
    echo "    Installed."
fi

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

echo "    Accepting conda TOS..."
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
echo "    TOS accepted."

# -------------------------------------------------------------------------------
# 2. Python Environment
# -------------------------------------------------------------------------------
echo ""
echo "[2/6] Python 3.13 environment..."

if conda env list | grep -q "^golf "; then
    echo "    Environment 'golf' already exists -- skipping."
else
    conda create -n golf python=3.13 -y
    echo "    Created."
fi

conda activate golf
echo "    Activated."

# -------------------------------------------------------------------------------
# 3. Requirements
# -------------------------------------------------------------------------------
echo ""
echo "[3/6] Requirements..."

if python3 -c "import torch, sentencepiece, numpy" 2>/dev/null; then
    echo "    Core packages already installed -- skipping."
else
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo "    Installed."
fi

# -------------------------------------------------------------------------------
# 4. FlashAttention-3 (MUST be FA3, not FA2)
# -------------------------------------------------------------------------------
echo ""
echo "[4/6] FlashAttention-3 (Hopper)..."

if python3 -c "import flash_attn_interface" 2>/dev/null; then
    echo "    FA3 already installed -- skipping."
elif python3 -c "import flash_attn; v=flash_attn.__version__; assert v.startswith('3')" 2>/dev/null; then
    echo "    FA3 already installed (flash_attn v3) -- skipping."
else
    echo "    Installing FA3 abi3 wheel..."
    pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
    echo "    Installed."
fi

# -------------------------------------------------------------------------------
# 5. zstandard (CRITICAL: prevents artifact size inflation)
# -------------------------------------------------------------------------------
echo ""
echo "[5/6] zstandard..."

if python3 -c "import zstandard" 2>/dev/null; then
    echo "    Already installed -- skipping."
else
    pip install zstandard -q
    echo "    Installed."
fi

# -------------------------------------------------------------------------------
# 6. Dataset (sp1024 for B-wing)
# -------------------------------------------------------------------------------
echo ""
echo "[6/6] FineWeb dataset (sp1024)..."

TRAIN_COUNT=$(ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
if [ "$TRAIN_COUNT" -ge 10 ]; then
    echo "    Already have $TRAIN_COUNT train shards -- skipping."
else
    echo "    Downloading... ($TRAIN_COUNT/80+ train shards found)"
    hf download sproos/parameter-golf-tokenizers --include "datasets/fineweb10B_sp1024/*" --local-dir ./data
    echo "    Downloaded."
fi

# -------------------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------------------
echo ""
echo "----------------------------------------------"
echo " Verification"
echo "----------------------------------------------"

python3 - << 'PYEOF'
import sys
import torch
import numpy as np
import glob

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}      : {props.name} ({props.total_memory // 1024**3}GB)")

fa_version = "NOT found"
try:
    import flash_attn_interface
    fa_version = "FA3 (flash_attn_interface)"
except ImportError:
    try:
        import flash_attn
        fa_version = f"{flash_attn.__version__}"
        if not fa_version.startswith("3"):
            fa_version += " WARNING: FA2 detected, need FA3!"
    except ImportError:
        pass
print(f"FlashAttn    : {fa_version}")

try:
    import zstandard
    print(f"zstandard    : OK")
except ImportError:
    print(f"zstandard    : MISSING -- artifact will inflate!")

train_files = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))
val_files   = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
print(f"Train shards : {len(train_files)}")
print(f"Val shards   : {len(val_files)}")

if val_files:
    total = sum(
        int(np.fromfile(f, dtype='<i4', count=3)[2])
        for f in val_files
    )
    print(f"Val tokens   : {total:,}")
PYEOF

echo ""
echo "----------------------------------------------"
echo " Done. Run B-wing IV with:"
echo "   conda activate golf"
echo "   bash experiments/B_wing/bwing_IV/run.sh"
echo "----------------------------------------------"
