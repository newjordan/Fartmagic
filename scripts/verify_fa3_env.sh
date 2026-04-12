#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VERIFY_DATA="${VERIFY_DATA:-1}"
FA3_VERIFY_ACTIVATE="${FA3_VERIFY_ACTIVATE:-0}"

if [[ "${FA3_VERIFY_ACTIVATE}" == "1" ]] && [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-fa3wheel}" >/dev/null 2>&1 || true
elif [[ "${FA3_VERIFY_ACTIVATE}" == "1" ]] && [[ -f "${VENV_DIR:-/workspace/venv_fa3}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR:-/workspace/venv_fa3}/bin/activate"
fi

RUNTIME_LIB_PATHS="$(python - <<'PYEOF'
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

python - <<'PYEOF'
import glob
import importlib
import os
import torch

torch_version = torch.__version__
torch_base = torch_version.split("+", 1)[0]
torch_parts = torch_base.split(".")
try:
    torch_mm = tuple(int(x) for x in torch_parts[:2])
except ValueError as exc:
    raise AssertionError(f"unparseable torch version: {torch_version}") from exc

assert torch_mm >= (2, 4), f"torch too old: {torch_version} (need >=2.4.x)"
cuda_version = str(torch.version.cuda or "")
try:
    cuda_major = int(cuda_version.split(".", 1)[0])
except (TypeError, ValueError):
    cuda_major = -1
assert cuda_major >= 12, f"wrong cuda: {torch.version.cuda} (need >=12.x)"

fa_ok = False
fa_errors = []
try:
    importlib.import_module("flash_attn_3._C")
    from flash_attn_interface import flash_attn_func  # noqa: F401
    fa_ok = True
except Exception as exc:
    fa_errors.append(f"flash_attn_3 path: {exc}")

if not fa_ok:
    try:
        from flash_attn_interface import flash_attn_func  # noqa: F401
        fa_ok = True
    except Exception as exc:
        fa_errors.append(f"flash_attn_interface path: {exc}")

if not fa_ok:
    try:
        import flash_attn
        assert str(getattr(flash_attn, "__version__", "")).startswith("3"), (
            f"flash_attn is not v3: {getattr(flash_attn, '__version__', '?')}"
        )
        fa_ok = True
    except Exception as exc:
        fa_errors.append(f"flash_attn package path: {exc}")

assert fa_ok, "flash-attn import failed; " + " | ".join(fa_errors)

verify_data = os.environ.get("VERIFY_DATA", "1") != "0"
variant = os.environ.get("DATASET_VARIANT", "sp1024")
tokenizer_map = {
    "sp1024": "./data/tokenizers/fineweb_1024_bpe.model",
    "sp8192": "./data/tokenizers/fineweb_8192_bpe.model",
}
dataset_dir = "fineweb10B_byte260" if variant == "byte260" else f"fineweb10B_{variant}"
train = []
val = []
if verify_data:
    tokenizer = os.environ.get("TOKENIZER_PATH", tokenizer_map.get(variant, ""))
    if tokenizer:
        assert os.path.isfile(tokenizer), f"missing tokenizer: {tokenizer}"

    train = glob.glob(f"./data/datasets/{dataset_dir}/fineweb_train_*.bin")
    val = glob.glob(f"./data/datasets/{dataset_dir}/fineweb_val_*.bin")
    assert len(train) >= 1, f"missing train shards for {dataset_dir}"
    assert len(val) >= 1, f"missing val shards for {dataset_dir}"

print("VERIFY_OK")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"gpus={torch.cuda.device_count()}")
if verify_data:
    print(f"dataset={dataset_dir} train_shards={len(train)} val_shards={len(val)}")
else:
    print("data_checks=skipped")
PYEOF
