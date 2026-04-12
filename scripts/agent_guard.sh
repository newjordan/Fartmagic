#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/agent_guard.sh status
  bash scripts/agent_guard.sh check <allowlist_file> [--delta-latest | --delta-from <snapshot_dir_or_paths_file>]
  bash scripts/agent_guard.sh snapshot [label] [--copy-untracked]

Allowlist format:
  - One path or prefix per line
  - Prefixes must end with /
  - Lines starting with # are ignored
USAGE
}

die() {
  echo "FATAL: $*" >&2
  exit 1
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

normalize_path() {
  local p="$1"
  p="${p#./}"
  printf '%s' "$p"
}

parse_status_paths() {
  git status --porcelain=v1 | while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local path="${line:3}"
    if [[ "$path" == *" -> "* ]]; then
      path="${path##* -> }"
    fi
    normalize_path "$path"
    printf '\n'
  done
}

is_allowed() {
  local path="$1"
  shift
  local pattern=""
  for pattern in "$@"; do
    if [[ "$pattern" == */ ]]; then
      [[ "$path" == "$pattern"* ]] && return 0
      continue
    fi
    if [[ "$pattern" == *"*"* || "$pattern" == *"?"* || "$pattern" == *"["* ]]; then
      [[ "$path" == $pattern ]] && return 0
      continue
    fi
    [[ "$path" == "$pattern" ]] && return 0
  done
  return 1
}

cmd_status() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    die "not inside a git repository"
  fi
  mapfile -t changed < <(parse_status_paths)
  if [[ ${#changed[@]} -eq 0 ]]; then
    echo "OK: working tree clean"
    return 0
  fi
  echo "Changed files (${#changed[@]}):"
  printf '  %s\n' "${changed[@]}"
}

cmd_check() {
  local allowlist_file="${1:-}"
  [[ -n "$allowlist_file" ]] || die "check requires <allowlist_file>"
  [[ -f "$allowlist_file" ]] || die "allowlist file not found: $allowlist_file"
  shift || true

  local baseline_paths_file=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --delta-latest)
        [[ -f ".agent_state/latest_snapshot_dir" ]] || die "no latest snapshot recorded; run snapshot first"
        local latest_dir
        latest_dir="$(cat .agent_state/latest_snapshot_dir)"
        baseline_paths_file="${latest_dir}/paths.txt"
        shift
        ;;
      --delta-from)
        [[ $# -ge 2 ]] || die "--delta-from requires <snapshot_dir_or_paths_file>"
        local baseline_ref="$2"
        if [[ -d "$baseline_ref" ]]; then
          baseline_paths_file="${baseline_ref}/paths.txt"
        else
          baseline_paths_file="$baseline_ref"
        fi
        shift 2
        ;;
      *)
        die "unknown check argument: $1"
        ;;
    esac
  done

  if [[ -n "$baseline_paths_file" ]]; then
    [[ -f "$baseline_paths_file" ]] || die "baseline paths file not found: $baseline_paths_file"
  fi

  mapfile -t changed < <(parse_status_paths)
  if [[ -n "$baseline_paths_file" ]]; then
    declare -A baseline=()
    local prior=""
    while IFS= read -r prior || [[ -n "$prior" ]]; do
      prior="$(trim "$prior")"
      [[ -z "$prior" ]] && continue
      baseline["$(normalize_path "$prior")"]=1
    done < "$baseline_paths_file"

    local delta=()
    local p=""
    for p in "${changed[@]}"; do
      if [[ -z "${baseline[$p]:-}" ]]; then
        delta+=("$p")
      fi
    done
    changed=("${delta[@]}")
  fi

  if [[ ${#changed[@]} -eq 0 ]]; then
    if [[ -n "$baseline_paths_file" ]]; then
      echo "OK: no new changed files since baseline (${baseline_paths_file})"
    else
      echo "OK: no changed files"
    fi
    return 0
  fi

  local patterns=()
  local raw=""
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    raw="$(trim "$raw")"
    [[ -z "$raw" ]] && continue
    [[ "$raw" == \#* ]] && continue
    patterns+=("$(normalize_path "$raw")")
  done < "$allowlist_file"

  [[ ${#patterns[@]} -gt 0 ]] || die "allowlist is empty: $allowlist_file"

  local violations=()
  local p=""
  for p in "${changed[@]}"; do
    if ! is_allowed "$p" "${patterns[@]}"; then
      violations+=("$p")
    fi
  done

  if [[ ${#violations[@]} -gt 0 ]]; then
    echo "FAIL: out-of-scope file changes detected:"
    printf '  %s\n' "${violations[@]}"
    echo "Allowlist: ${allowlist_file}"
    return 2
  fi

  echo "OK: all changed files are in allowlist (${allowlist_file})"
}

cmd_snapshot() {
  local label="manual"
  local copy_untracked="0"

  if [[ $# -gt 0 && "$1" != --* ]]; then
    label="$1"
    shift
  fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --copy-untracked)
        copy_untracked="1"
        shift
        ;;
      *)
        die "unknown snapshot argument: $1"
        ;;
    esac
  done

  local safe_label
  safe_label="$(printf '%s' "$label" | tr '[:space:]' '_' | sed -E 's/[^a-zA-Z0-9._-]+/_/g')"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local out_dir=".agent_state/snapshots/${ts}_${safe_label}"

  mkdir -p "${out_dir}"
  git status --porcelain=v1 -b > "${out_dir}/status.txt"
  parse_status_paths > "${out_dir}/paths.txt"
  git diff --binary > "${out_dir}/tracked.diff"
  git diff --cached --binary > "${out_dir}/staged.diff"
  git ls-files --others --exclude-standard > "${out_dir}/untracked_files.txt"

  if [[ "$copy_untracked" == "1" ]]; then
    mkdir -p "${out_dir}/untracked"
    while IFS= read -r file_path || [[ -n "$file_path" ]]; do
      [[ -z "$file_path" ]] && continue
      if [[ -f "$file_path" ]]; then
        mkdir -p "${out_dir}/untracked/$(dirname "$file_path")"
        cp -p "$file_path" "${out_dir}/untracked/${file_path}"
      fi
    done < "${out_dir}/untracked_files.txt"
  fi

  mkdir -p ".agent_state"
  printf '%s\n' "${out_dir}" > ".agent_state/latest_snapshot_dir"

  echo "Snapshot created: ${out_dir}"
  echo "  - status.txt"
  echo "  - paths.txt"
  echo "  - tracked.diff"
  echo "  - staged.diff"
  echo "  - untracked_files.txt"
  if [[ "$copy_untracked" == "1" ]]; then
    echo "  - untracked/ (copied)"
  else
    echo "  - untracked/ not copied (use --copy-untracked to include)"
  fi
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    status)
      shift
      cmd_status "$@"
      ;;
    check)
      shift
      cmd_check "$@"
      ;;
    snapshot)
      shift
      cmd_snapshot "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      [[ -n "$cmd" ]] && echo "" >&2 && echo "Unknown command: $cmd" >&2
      exit 1
      ;;
  esac
}

main "$@"
