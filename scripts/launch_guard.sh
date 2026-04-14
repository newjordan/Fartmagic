#!/usr/bin/env bash
set -euo pipefail

lg_die() {
  echo "FATAL[launch_guard]: $*" >&2
  exit 1
}

lg_require_path() {
  local p="$1"
  local label="${2:-path}"
  if [[ ! -e "$p" ]]; then
    lg_die "missing ${label}: ${p}"
  fi
}

lg_verify_pod_stack_lock() {
  local repo_root="$1"
  local guard="${repo_root}/scripts/pod_stack_guard.sh"
  if [[ ! -x "$guard" ]]; then
    lg_die "pod stack guard missing or not executable: ${guard}"
  fi
  if [[ "${SKIP_POD_STACK_GUARD:-0}" == "1" ]]; then
    echo "[launch_guard] WARNING: SKIP_POD_STACK_GUARD=1 bypasses pod setup lock verification"
    return 0
  fi
  bash "$guard" verify >/dev/null
  echo "[launch_guard] pod_stack_lock verified"
}

lg_visible_gpus() {
  local n=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$n" =~ ^[0-9]+$ ]]; then
      echo "$n"
      return 0
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    n="$(python3 - <<'PY'
try:
    import torch
    print(int(torch.cuda.device_count()))
except Exception:
    print(-1)
PY
)"
    if [[ "$n" =~ ^-?[0-9]+$ ]]; then
      echo "$n"
      return 0
    fi
  fi

  echo "-1"
}

lg_assert_world_size() {
  local nproc="$1"
  [[ "$nproc" =~ ^[0-9]+$ ]] || lg_die "NPROC_PER_NODE must be an integer, got: ${nproc}"
  (( nproc >= 1 )) || lg_die "NPROC_PER_NODE must be >= 1, got: ${nproc}"

  local gpus
  gpus="$(lg_visible_gpus)"

  if [[ "$gpus" =~ ^[0-9]+$ ]]; then
    if (( gpus < nproc )); then
      lg_die "world-size mismatch: visible_gpus=${gpus} < NPROC_PER_NODE=${nproc}"
    fi
    echo "[launch_guard] gpu_check visible_gpus=${gpus} nproc_per_node=${nproc}"
    return 0
  fi

  if (( nproc > 1 )); then
    lg_die "cannot verify GPU count, refusing multi-process launch (NPROC_PER_NODE=${nproc})"
  fi
  echo "[launch_guard] gpu_check unknown GPU count, allowing nproc_per_node=1"
}

lg_lock_release() {
  if [[ -n "${LG_LOCK_DIR:-}" && -d "${LG_LOCK_DIR}" ]]; then
    rm -rf "${LG_LOCK_DIR}" || true
  fi
}

lg_lock_acquire() {
  local repo_root="$1"
  local train_script="$2"
  local scope="${repo_root}::${train_script}"
  local key
  key="$(printf '%s' "$scope" | sha256sum | awk '{print $1}')"

  LG_LOCK_DIR="/tmp/sota_launch_guard_${key}.lock"
  local pid_file="${LG_LOCK_DIR}/pid"
  local meta_file="${LG_LOCK_DIR}/meta"

  if mkdir "${LG_LOCK_DIR}" 2>/dev/null; then
    printf '%s\n' "$$" > "${pid_file}"
    printf '%s\n' "$scope" > "${meta_file}"
    trap lg_lock_release EXIT INT TERM
    echo "[launch_guard] lock_acquired ${LG_LOCK_DIR} pid=$$"
    return 0
  fi

  local owner_pid=""
  if [[ -f "${pid_file}" ]]; then
    owner_pid="$(cat "${pid_file}" 2>/dev/null || true)"
  fi

  if [[ -n "$owner_pid" ]] && ps -p "$owner_pid" >/dev/null 2>&1; then
    lg_die "active run lock exists at ${LG_LOCK_DIR} (owner pid=${owner_pid}); stop/finish that run before relaunch"
  fi

  rm -rf "${LG_LOCK_DIR}" || lg_die "stale lock exists and cannot be removed: ${LG_LOCK_DIR}"
  mkdir "${LG_LOCK_DIR}" || lg_die "cannot acquire run lock after stale cleanup: ${LG_LOCK_DIR}"
  printf '%s\n' "$$" > "${pid_file}"
  printf '%s\n' "$scope" > "${meta_file}"
  trap lg_lock_release EXIT INT TERM
  echo "[launch_guard] lock_recovered ${LG_LOCK_DIR} pid=$$"
}

lg_preflight_and_lock() {
  local repo_root="$1"
  local train_script="$2"
  local tracked_env="$3"
  local nproc="$4"
  local invoker="${5:-unknown}"

  echo "[launch_guard] intent invoker=${invoker} repo_root=${repo_root} train_script=${train_script} nproc=${nproc}"

  lg_require_path "$repo_root" "repo_root"
  lg_require_path "$train_script" "train_script"
  lg_require_path "$tracked_env" "tracked_env"
  lg_require_path "${repo_root}/scripts/leg_diff_guard.py" "leg_diff_guard"
  lg_verify_pod_stack_lock "$repo_root"

  command -v torchrun >/dev/null 2>&1 || lg_die "torchrun not found in PATH"

  if [[ -n "${DATA_PATH:-}" ]]; then
    lg_require_path "${DATA_PATH}" "DATA_PATH"
  fi
  if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    lg_require_path "${TOKENIZER_PATH}" "TOKENIZER_PATH"
  fi

  lg_assert_world_size "$nproc"
  lg_lock_acquire "$repo_root" "$train_script"
}
