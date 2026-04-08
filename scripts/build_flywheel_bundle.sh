#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_WHEEL="${REPO_ROOT}/wheels/fa3_cu124_vast/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl"
WHEEL_PATH="${WHEEL_PATH:-${DEFAULT_WHEEL}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/dist}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_NAME="flywheel_cu124_fa3_${STAMP}"
BUNDLE_DIR="${OUT_DIR}/${BUNDLE_NAME}"

if [[ ! -f "${WHEEL_PATH}" ]]; then
  echo "FATAL: wheel not found: ${WHEEL_PATH}"
  echo "Set WHEEL_PATH=/abs/path/to/flash_attn_3-*.whl"
  exit 1
fi

mkdir -p "${BUNDLE_DIR}/scripts"

cp -f "${WHEEL_PATH}" "${BUNDLE_DIR}/"
WHEEL_BASENAME="$(basename -- "${WHEEL_PATH}")"
if [[ -f "${WHEEL_PATH}.sha256" ]]; then
  cp -f "${WHEEL_PATH}.sha256" "${BUNDLE_DIR}/"
else
  (cd "${BUNDLE_DIR}" && sha256sum "${WHEEL_BASENAME}" > "${WHEEL_BASENAME}.sha256")
fi

cp -f "${REPO_ROOT}/scripts/run_rascal_slot_locked.sh" "${BUNDLE_DIR}/scripts/"
cp -f "${REPO_ROOT}/scripts/verify_cu124_fa3_env.sh" "${BUNDLE_DIR}/scripts/"
chmod +x "${BUNDLE_DIR}/scripts/run_rascal_slot_locked.sh" "${BUNDLE_DIR}/scripts/verify_cu124_fa3_env.sh"

cat > "${BUNDLE_DIR}/install_flywheel.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"
CONDA_ENV="${CONDA_ENV:-fa3wheel}"
VENV_DIR="${VENV_DIR:-/workspace/venv_cu124}"
WHEEL_PATH="$(ls -1 "${SELF_DIR}"/flash_attn_3-*.whl | head -n1)"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "FATAL: repo dir missing: ${REPO_DIR}"
  exit 1
fi

if [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}" || conda create -y -n "${CONDA_ENV}" python=3.12 pip
  conda activate "${CONDA_ENV}"
else
  [[ -d "${VENV_DIR}" ]] || python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124
python -m pip install \
  sentencepiece zstandard huggingface-hub datasets tiktoken attr einops ninja packaging sympy==1.12
python -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"

mkdir -p "${REPO_DIR}/scripts"
cp -f "${SELF_DIR}/scripts/run_rascal_slot_locked.sh" "${REPO_DIR}/scripts/run_rascal_slot_locked.sh"
cp -f "${SELF_DIR}/scripts/verify_cu124_fa3_env.sh" "${REPO_DIR}/scripts/verify_cu124_fa3_env.sh"
chmod +x "${REPO_DIR}/scripts/run_rascal_slot_locked.sh" "${REPO_DIR}/scripts/verify_cu124_fa3_env.sh"

cat > "${REPO_DIR}/scripts/activate_flywheel_env.sh" <<ACTEOF
#!/usr/bin/env bash
set -euo pipefail
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
export COMPILE_ENABLED=1
export COMPILE_FULLGRAPH=1
export TORCHDYNAMO_SUPPRESS_ERRORS=0
ACTEOF
chmod +x "${REPO_DIR}/scripts/activate_flywheel_env.sh"

bash "${REPO_DIR}/scripts/verify_cu124_fa3_env.sh"
echo "READY: flywheel installed."
echo "Next: source ${REPO_DIR}/scripts/activate_flywheel_env.sh"
echo "Run : cd ${REPO_DIR} && SEED=300 NPROC_PER_NODE=8 bash scripts/run_rascal_slot_locked.sh"
EOF
chmod +x "${BUNDLE_DIR}/install_flywheel.sh"

cat > "${BUNDLE_DIR}/README.txt" <<EOF
Flywheel Bundle (cu124 + FA3 + locked settings)
===============================================

Contents:
- ${WHEEL_BASENAME}
- ${WHEEL_BASENAME}.sha256
- install_flywheel.sh
- scripts/run_rascal_slot_locked.sh
- scripts/verify_cu124_fa3_env.sh

On target pod:
1) cd /workspace
2) tar -xzf ${BUNDLE_NAME}.tar.gz
3) cd ${BUNDLE_NAME}
4) bash install_flywheel.sh
5) source /workspace/parameter-golf/scripts/activate_flywheel_env.sh
6) cd /workspace/parameter-golf && SEED=300 NPROC_PER_NODE=8 bash scripts/run_rascal_slot_locked.sh
EOF

(
  cd "${OUT_DIR}"
  tar -czf "${BUNDLE_NAME}.tar.gz" "${BUNDLE_NAME}"
)

echo "BUNDLE_DIR=${BUNDLE_DIR}"
echo "BUNDLE_TAR=${OUT_DIR}/${BUNDLE_NAME}.tar.gz"
echo "READY: flywheel bundle built."

