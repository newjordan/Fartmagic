#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${REPO_ROOT}/scripts/pod_stack.lock"

# Files that must remain stable for pod setup correctness.
PROTECTED_FILES=(
  "scripts/Im_sorry_pod_setup.sh"
)

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/pod_stack_guard.sh verify
  bash scripts/pod_stack_guard.sh status
  bash scripts/pod_stack_guard.sh update

Commands:
  verify  Fail closed if protected files differ from lock.
  status  Show current hash status without failing.
  update  Rewrite lock file from current file hashes (intentional only).
USAGE
}

die() {
  echo "FATAL[pod_stack_guard]: $*" >&2
  exit 1
}

require_tools() {
  command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"
}

write_lock() {
  require_tools
  : > "${LOCK_FILE}"
  local rel=""
  local abs=""
  local h=""
  for rel in "${PROTECTED_FILES[@]}"; do
    abs="${REPO_ROOT}/${rel}"
    [[ -f "${abs}" ]] || die "missing protected file: ${rel}"
    h="$(sha256sum "${abs}" | awk '{print $1}')"
    printf '%s\t%s\n' "${rel}" "${h}" >> "${LOCK_FILE}"
  done
  echo "[pod_stack_guard] lock updated: ${LOCK_FILE}"
}

verify_lock_loaded() {
  [[ -f "${LOCK_FILE}" ]] || die "lock file missing: ${LOCK_FILE} (run: bash scripts/pod_stack_guard.sh update)"
}

check_one() {
  local rel="$1"
  local expected="$2"
  local abs="${REPO_ROOT}/${rel}"
  [[ -f "${abs}" ]] || { echo "MISSING\t${rel}\t-\t${expected}"; return 2; }
  local actual
  actual="$(sha256sum "${abs}" | awk '{print $1}')"
  if [[ "${actual}" == "${expected}" ]]; then
    echo "OK\t${rel}\t${actual}\t${expected}"
    return 0
  fi
  echo "MISMATCH\t${rel}\t${actual}\t${expected}"
  return 1
}

verify_or_status() {
  local mode="$1" # verify|status
  require_tools
  verify_lock_loaded
  local line=""
  local rel=""
  local expected=""
  local rc=0
  while IFS=$'\t' read -r rel expected || [[ -n "${rel:-}" ]]; do
    [[ -z "${rel:-}" ]] && continue
    [[ "${rel}" == \#* ]] && continue
    check_one "${rel}" "${expected}" || rc=1
  done < "${LOCK_FILE}"

  if [[ "${mode}" == "verify" && ${rc} -ne 0 ]]; then
    die "pod stack lock mismatch detected; refuse to continue"
  fi
  return 0
}

main() {
  local cmd="${1:-verify}"
  case "${cmd}" in
    verify)
      verify_or_status verify
      ;;
    status)
      verify_or_status status
      ;;
    update)
      write_lock
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      die "unknown command: ${cmd}"
      ;;
  esac
}

main "$@"
