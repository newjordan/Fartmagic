#!/bin/bash
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore   # suppress "running as root" pip warning
# =============================================================================
# POD SETUP — dated to the Midnight 12L PR window
#
# Usage:  bash scripts/Im_sorry_pod_setup.sh
#   (or curl from raw URL and pipe to bash — works either way)
#
# Anchor:
#   OpenAI PR #1458 "Midnight 12L"
#   Created 2026-04-07 19:40:32 America/Chicago
#
# What it does:
#   1. Clones/syncs repo to the TEST_LAB branch
#   2. Installs deps (pip, zstandard, FA3, dataset)
#   3. Verifies everything works
#   4. Done. You run your experiment manually.
# =============================================================================

REPO_URL="https://github.com/newjordan/parameter-golf.git"
BRANCH="${BRANCH:-TEST_LAB}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
DATASET_VARIANT="${DATASET_VARIANT:-sp1024}"
ACTIVATE_HELPER_REL="scripts/activate_pod_env.sh"
# Auto-detect repo root from script location; fall back for curl-pipe scenario
_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)" || true
_CANDIDATE="$(cd -- "${_SCRIPT_DIR}/.." && pwd 2>/dev/null)" || true
if [[ -d "${_CANDIDATE}/.git" ]]; then
    WORKSPACE="${_CANDIDATE}"
else
    WORKSPACE="/workspace/parameter-golf"
fi
LOCAL_FA3_WHEEL_REL="wheels/fa3_vast/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl"
LOCAL_FA3_WHEEL="${LOCAL_FA3_WHEEL:-${WORKSPACE}/${LOCAL_FA3_WHEEL_REL}}"

build_runtime_lib_path() {
    python3 - <<'PYEOF'
import glob
import os
import site
import torch

paths = []
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(torch_lib):
    paths.append(torch_lib)

site_dirs = []
try:
    site_dirs.extend(site.getsitepackages())
except Exception:
    pass
try:
    site_dirs.append(site.getusersitepackages())
except Exception:
    pass

for base in site_dirs:
    if not base or not os.path.isdir(base):
        continue
    for pattern in (
        os.path.join(base, "nvidia", "*", "lib"),
        os.path.join(base, "nvidia", "*", "lib64"),
    ):
        for candidate in glob.glob(pattern):
            if os.path.isdir(candidate):
                paths.append(candidate)

for candidate in (
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/compat",
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
):
    if os.path.isdir(candidate):
        paths.append(candidate)

seen = set()
ordered = []
for path in paths:
    if path not in seen:
        seen.add(path)
        ordered.append(path)
print(":".join(ordered))
PYEOF
}

echo "============================================"
echo "  POD SETUP"
echo "  Branch: ${BRANCH}"
echo "  Variant: ${DATASET_VARIANT}"
echo "  Train shards: ${TRAIN_SHARDS}"
echo "  Anchor: Midnight 12L PR #1458"
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
RUNTIME_LIB_PATHS="$(build_runtime_lib_path)"
export LD_LIBRARY_PATH="${RUNTIME_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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

ACTIVATE_HELPER="${WORKSPACE}/${ACTIVATE_HELPER_REL}"
cat > "${ACTIVATE_HELPER}" <<'ACTEOF'
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_LIB_PATHS="$(python3 - <<'PYEOF'
import glob
import os
import site
import torch

paths = []
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(torch_lib):
    paths.append(torch_lib)

site_dirs = []
try:
    site_dirs.extend(site.getsitepackages())
except Exception:
    pass
try:
    site_dirs.append(site.getusersitepackages())
except Exception:
    pass

for base in site_dirs:
    if not base or not os.path.isdir(base):
        continue
    for pattern in (
        os.path.join(base, "nvidia", "*", "lib"),
        os.path.join(base, "nvidia", "*", "lib64"),
    ):
        for candidate in glob.glob(pattern):
            if os.path.isdir(candidate):
                paths.append(candidate)

for candidate in (
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/compat",
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
):
    if os.path.isdir(candidate):
        paths.append(candidate)

seen = set()
ordered = []
for path in paths:
    if path not in seen:
        seen.add(path)
        ordered.append(path)
print(":".join(ordered))
PYEOF
)"
export LD_LIBRARY_PATH="${RUNTIME_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -d "${REPO_ROOT}/flash-attention/hopper" ]]; then
    export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
fi
ACTEOF
chmod +x "${ACTIVATE_HELPER}"
echo "  Wrote ${ACTIVATE_HELPER_REL}"

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

link_flash_attn_config() {
    python3 - <<'PYEOF' >/dev/null 2>&1 || true
import inspect
import os
import site

import flash_attn_interface

cfg_src = os.path.join(os.path.dirname(inspect.getfile(flash_attn_interface)), "flash_attn_config.py")
sp = site.getsitepackages()[0]
cfg_dst = os.path.join(sp, "flash_attn_config.py")
if os.path.isfile(cfg_src) and not os.path.exists(cfg_dst):
    os.symlink(cfg_src, cfg_dst)
PYEOF
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

verify_fa3_runtime() {
    python3 - <<'PYEOF' >/dev/null 2>&1
import importlib
import flash_attn_interface  # noqa: F401
importlib.import_module("flash_attn_3._C")
PYEOF
}

describe_fa3_failure() {
    python3 - <<'PYEOF' 2>/dev/null || true
import importlib

errors = []
try:
    import flash_attn_interface  # noqa: F401
except Exception as exc:
    errors.append(f"flash_attn_interface: {exc}")

try:
    importlib.import_module("flash_attn_3._C")
except Exception as exc:
    errors.append(f"flash_attn_3._C: {exc}")

if not errors:
    print("  FA3 import failure reason unavailable")
else:
    for err in errors:
        print(f"  {err}")
PYEOF
}

local_wheel_looks_compatible() {
    python3 - <<'PYEOF' >/dev/null 2>&1
import torch

tv = torch.__version__
cv = str(torch.version.cuda or "")
if not tv.startswith("2.4.1"):
    raise SystemExit(1)
if not cv.startswith("12.4"):
    raise SystemExit(1)
PYEOF
}

install_hopper_fast() {
    local hopper_dir="$1"
    if [ ! -f "${hopper_dir}/setup.py" ] && [ ! -f "${hopper_dir}/pyproject.toml" ]; then
        return 1
    fi

    echo "  Building local Hopper FA3 with historical fast path..."
    if (
        cd "${hopper_dir}"
        export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
        export FLASH_ATTENTION_DISABLE_FP8=TRUE
        export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
        export FLASH_ATTENTION_DISABLE_SM80=TRUE
        export MAX_JOBS="${MAX_JOBS:-4}"
        export TMPDIR="${TMPDIR:-/workspace/tmp}"
        mkdir -p "${TMPDIR}"
        python3 -m pip install -U ninja packaging >/dev/null
        python3 -m pip install -e . --no-build-isolation
    ) 2>&1 | tail -20; then
        if verify_fa3_runtime; then
            link_flash_attn_config
            echo "  Local Hopper FA3 built and attached"
            return 0
        fi
        echo "  Local Hopper fast install completed but did not import cleanly:"
        describe_fa3_failure
        return 1
    fi

    echo "  Local Hopper fast install failed; continuing..."
    return 1
}

install_fa3() {
    echo "  Searching system for pre-installed flash_attn_interface..."
    local system_fa3_dir=""
    system_fa3_dir="$(find_system_fa3_dir || true)"
    if [ -n "${system_fa3_dir}" ] && sync_fa3_dir_into_site "${system_fa3_dir}"; then
        if verify_fa3_runtime; then
            echo "  Attached system FA3 runtime"
            return 0
        fi
        echo "  System FA3 path did not import cleanly; continuing..."
    fi

    echo "  Checking for local flash-attention/hopper source..."
    if [ -d "${WORKSPACE}/flash-attention/hopper" ]; then
        if install_hopper_fast "${WORKSPACE}/flash-attention/hopper"; then
            return 0
        fi
        if sync_fa3_dir_into_site "${WORKSPACE}/flash-attention/hopper"; then
            if verify_fa3_runtime; then
                link_flash_attn_config
                echo "  Local Hopper FA3 attached"
                return 0
            fi
            echo "  Local Hopper attach did not import cleanly; continuing..."
            describe_fa3_failure
        fi
    fi

    echo "  Checking for repo-local emergency FA3 wheel..."
    if [ -f "${LOCAL_FA3_WHEEL}" ]; then
        if local_wheel_looks_compatible; then
            if python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
                "${LOCAL_FA3_WHEEL}" 2>&1 | tail -3; then
                if verify_fa3_runtime; then
                    link_flash_attn_config
                    echo "  Local emergency wheel attached (${LOCAL_FA3_WHEEL_REL})"
                    return 0
                fi
                echo "  Local emergency wheel installed but did not import cleanly; continuing..."
                describe_fa3_failure
            fi
        else
            echo "  Skipping repo-local emergency wheel on this torch/CUDA stack"
        fi
    else
        echo "  No repo-local emergency wheel at ${LOCAL_FA3_WHEEL_REL}"
    fi

    echo "  Attempting FA3 abi3 wheel (cu128)..."
    if python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
        "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        2>&1 | tail -3; then
        if verify_fa3_runtime; then
            link_flash_attn_config
            echo "  Direct abi wheel attached"
            return 0
        fi
        echo "  Direct abi wheel installed but did not import cleanly."
        describe_fa3_failure
    fi

    echo "  WARNING: Could not install FA3. Will fall back to PyTorch SDPA."
    return 1
}

if python3 -c "from flash_attn_interface import flash_attn_func; print('  FA3 (flash_attn_interface) OK')" 2>/dev/null; then
    : # already good
elif python3 -c "import flash_attn; v=flash_attn.__version__; assert v.startswith('3'); print(f'  FA3 v{v} OK')" 2>/dev/null; then
    : # flash_attn v3 package works
else
    install_fa3
fi

DATASET_VARIANT="${DATASET_VARIANT}" VERIFY_DATA=0 bash "${WORKSPACE}/scripts/verify_fa3_env.sh"

# =============================================================================
# 6. Dataset
# =============================================================================
echo ""
echo "[6/6] Tokenizer + FineWeb dataset (${DATASET_VARIANT})..."

# Use competition's official download script (willdepueoai/parameter-golf dataset repo)
# NOT sproos/parameter-golf-tokenizers — that repo has different val shard (58M vs 62M tokens)
echo "  Using competition download script (data/cached_challenge_fineweb.py)..."
cd "${WORKSPACE}"
python3 data/cached_challenge_fineweb.py --variant "${DATASET_VARIANT}" --train-shards "${TRAIN_SHARDS}"
echo "  Competition data downloaded"

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "============================================"
echo " Verification"
echo "============================================"

DATASET_VARIANT="${DATASET_VARIANT}" VERIFY_DATA=1 bash "${WORKSPACE}/scripts/verify_fa3_env.sh"

echo ""
echo "============================================"
echo " READY."
echo "============================================"
echo "Activation helper: ${ACTIVATE_HELPER_REL}"
