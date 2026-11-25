#!/usr/bin/env bash
# Resolve FineWeb sp8192 dataset and tokenizer paths.
# Existing DATA_PATH and TOKENIZER_PATH exports win.
# Candidates are repo-relative paths and shared pod locations.
# This file does not consult a user home directory.

fm_resolve_sp8192() {
  local script_dir="$1"
  local repo_root
  repo_root="$(cd -- "${script_dir}/.." && pwd)"

  if [[ -z "${DATA_PATH:-}" ]]; then
    local candidate=""
    for candidate in \
      "${repo_root}/data/datasets/fineweb10B_sp8192" \
      "${script_dir}/../../data/datasets/fineweb10B_sp8192" \
      "/dev/shm/fineweb10B_sp8192" \
      "/root/Fartmagic/data/datasets/fineweb10B_sp8192" \
      "/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192"
    do
      if [[ -d "${candidate}" ]]; then
        export DATA_PATH="${candidate}"
        break
      fi
    done
  fi

  if [[ -z "${TOKENIZER_PATH:-}" ]]; then
    local candidate=""
    for candidate in \
      "${repo_root}/data/tokenizers/fineweb_8192_bpe.model" \
      "${script_dir}/../../data/tokenizers/fineweb_8192_bpe.model" \
      "/dev/shm/fineweb_8192_bpe.model" \
      "/root/Fartmagic/data/tokenizers/fineweb_8192_bpe.model" \
      "/workspace/SOTA_FINAL/data/tokenizers/fineweb_8192_bpe.model"
    do
      if [[ -f "${candidate}" ]]; then
        export TOKENIZER_PATH="${candidate}"
        break
      fi
    done
  fi
}

fm_require_sp8192() {
  if [[ -z "${DATA_PATH:-}" || ! -d "${DATA_PATH}" ]]; then
    echo "FATAL: Cannot find fineweb10B_sp8192 dataset. Set DATA_PATH." >&2
    exit 1
  fi
  if [[ -z "${TOKENIZER_PATH:-}" || ! -f "${TOKENIZER_PATH}" ]]; then
    echo "FATAL: Cannot find fineweb_8192_bpe.model tokenizer. Set TOKENIZER_PATH." >&2
    exit 1
  fi
}
