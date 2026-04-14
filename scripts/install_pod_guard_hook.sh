#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_PATH="${REPO_ROOT}/.git/hooks/pre-commit"

mkdir -p "$(dirname "${HOOK_PATH}")"
cat > "${HOOK_PATH}" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_POD_STACK_EDIT:-0}" == "1" ]]; then
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  exit 0
fi

changed="$(git diff --cached --name-only -- scripts/Im_sorry_pod_setup.sh || true)"
if [[ -n "${changed}" ]]; then
  echo "FATAL[pre-commit]: pod stack files are protected." >&2
  echo "Blocked files:" >&2
  echo "${changed}" >&2
  echo "If intentional, commit with: ALLOW_POD_STACK_EDIT=1 git commit ..." >&2
  exit 1
fi
HOOK

chmod +x "${HOOK_PATH}"
echo "[pod_guard_hook] installed: ${HOOK_PATH}"
