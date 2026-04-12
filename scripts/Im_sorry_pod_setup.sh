#!/usr/bin/env bash
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore

# Midnight 12L parity anchor:
#   OpenAI PR #1458
#   Created: 2026-04-07 19:40:32 America/Chicago
#   Stack: torch 2.4.1+cu124 + local FA3 wheel + H100 Hopper path

REPO_URL="https://github.com/newjordan/parameter-golf.git"
BRANCH="${BRANCH:-TEST_LAB}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
CONDA_ENV="${CONDA_ENV:-fa3wheel}"
VENV_DIR="${VENV_DIR:-/workspace/venv_cu124}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)" || true
_CANDIDATE="$(cd -- "${_SCRIPT_DIR}/.." && pwd 2>/dev/null)" || true
if [[ -d "${_CANDIDATE}/.git" ]]; then
    WORKSPACE="${_CANDIDATE}"
else
    WORKSPACE="/workspace/parameter-golf"
fi

WHEEL_PATH="${WHEEL_PATH:-${WORKSPACE}/wheels/fa3_cu124_vast/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl}"
ACTIVATE_FLYWHEEL="${WORKSPACE}/scripts/activate_flywheel_env.sh"
ACTIVATE_POD="${WORKSPACE}/scripts/activate_pod_env.sh"

log() { printf '%s\n' "$*"; }
die() { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

activate_runtime_env() {
    if [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
        # shellcheck disable=SC1091
        source /workspace/miniconda3/etc/profile.d/conda.sh
        conda activate "${CONDA_ENV}"
    elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "${VENV_DIR}/bin/activate"
    else
        die "runtime env missing; expected conda env ${CONDA_ENV} or venv ${VENV_DIR}"
    fi
}

write_activate_helper() {
    local helper_path="$1"
    cat > "${helper_path}" <<ACTEOF
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi
TORCH_LIB=\$(python - <<'PYEOF'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYEOF
)
export LD_LIBRARY_PATH="\${TORCH_LIB}:\${LD_LIBRARY_PATH:-}"
if [[ -d "\${REPO_ROOT}/flash-attention/hopper" ]]; then
  export PYTHONPATH="\${REPO_ROOT}/flash-attention/hopper:\${PYTHONPATH:-}"
fi
for _site in /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/site-packages /usr/lib/python3/dist-packages; do
  if [[ -d "\${_site}" ]]; then
    export PYTHONPATH="\${_site}:\${PYTHONPATH:-}"
  fi
done
export COMPILE_ENABLED=1
export COMPILE_FULLGRAPH=1
export TORCHDYNAMO_SUPPRESS_ERRORS=0
ACTEOF
    chmod +x "${helper_path}"
}

log "============================================"
log "  POD SETUP"
log "  Branch: ${BRANCH}"
log "  Train shards: ${TRAIN_SHARDS}"
log "  Midnight anchor: PR #1458 @ 2026-04-07 19:40:32 America/Chicago"
log "============================================"

if [[ -d "${WORKSPACE}/.git" ]]; then
    log "[1/6] Repo exists, force-syncing to ${BRANCH}..."
    cd "${WORKSPACE}"
    git fetch origin "${BRANCH}" --quiet
    git checkout -B "${BRANCH}" "origin/${BRANCH}" --force
    git clean -fd --quiet
elif [[ -d "${WORKSPACE}" ]]; then
    log "[1/6] Existing non-git workspace detected, using in-place files..."
    cd "${WORKSPACE}"
else
    log "[1/6] Cloning repo..."
    git clone -b "${BRANCH}" "${REPO_URL}" "${WORKSPACE}"
    cd "${WORKSPACE}"
fi
if [[ -d "${WORKSPACE}/.git" ]]; then
    log "  HEAD: $(git log --oneline -1)"
else
    log "  HEAD: non-git workspace (no commit metadata)"
fi

[[ -f "${WHEEL_PATH}" ]] || die "missing FA3 wheel: ${WHEEL_PATH}"

log ""
log "[2/6] Base image forensics..."
python3 --version || die "python3 not found"
python3 - <<'PYEOF' || true
try:
    import torch
    print(f"  Base torch {torch.__version__}  CUDA {torch.version.cuda}")
except Exception as exc:
    print(f"  Base torch unavailable: {exc}")
PYEOF
log "  Installing isolated Midnight stack regardless of base image."

log ""
log "[3/6] Installing exact Midnight stack (torch 2.4.1+cu124 + FA3 wheel)..."
if [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /workspace/miniconda3/etc/profile.d/conda.sh
    if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
        log "  Creating conda env ${CONDA_ENV}"
        conda create -y -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" pip
    else
        log "  Reusing conda env ${CONDA_ENV}"
    fi
    conda activate "${CONDA_ENV}"
else
    log "  Reusing venv ${VENV_DIR}"
    if [[ ! -d "${VENV_DIR}" ]]; then
        python3 -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
fi

python -m pip install -U pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" \
    torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124
python -m pip install \
    sentencepiece zstandard brotli huggingface-hub datasets tiktoken attr einops ninja packaging sympy==1.12
python -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"

log ""
log "[4/6] Writing activation helpers..."
write_activate_helper "${ACTIVATE_FLYWHEEL}"
write_activate_helper "${ACTIVATE_POD}"
log "  Wrote scripts/activate_flywheel_env.sh"
log "  Wrote scripts/activate_pod_env.sh"

log ""
log "[5/6] Verifying FA3 + Midnight runtime..."
activate_runtime_env
bash "${WORKSPACE}/scripts/verify_cu124_fa3_env.sh"

log ""
log "[6/6] Downloading tokenizers + FineWeb datasets..."
activate_runtime_env
cd "${WORKSPACE}"
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python data/cached_challenge_fineweb.py --variant sp8192 --train-shards "${TRAIN_SHARDS}" || {
    log "  WARNING: SP8192 download failed; SP1024 is still ready."
}
rm -f data/manifest.json

log ""
log "============================================"
log " READY."
log "============================================"
log "Activation helpers:"
log "  scripts/activate_flywheel_env.sh"
log "  scripts/activate_pod_env.sh"
