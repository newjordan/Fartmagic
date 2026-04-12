#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/new_leg.sh <name> [--source <path_to_train_gpt.py>]

Examples:
  bash scripts/new_leg.sh qk_gain_525
  bash scripts/new_leg.sh vocab_trim --source vault/train_gpt_midnight_12l_sota_REAL.py
USAGE
}

die() {
  echo "FATAL: $*" >&2
  exit 1
}

slugify() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_{2,}/_/g'
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RAW_NAME="$1"
shift

SOURCE_REL="vault/train_gpt_midnight_iii_base.py"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      [[ $# -ge 2 ]] || die "--source requires a value"
      SOURCE_REL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

NAME_SLUG="$(slugify "${RAW_NAME}")"
[[ -n "${NAME_SLUG}" ]] || die "name produced empty slug; use letters/numbers"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATE_TAG="$(date +%F)"
LEG_DIR="legs/${DATE_TAG}_${NAME_SLUG}"

if [[ "${SOURCE_REL}" = /* ]]; then
  SOURCE_PATH="${SOURCE_REL}"
else
  SOURCE_PATH="${REPO_ROOT}/${SOURCE_REL}"
fi

PARENT_REF="${SOURCE_REL}"
if [[ "${SOURCE_PATH}" == "${REPO_ROOT}/"* ]]; then
  PARENT_REF="${SOURCE_PATH#${REPO_ROOT}/}"
fi
[[ -n "${PARENT_REF}" ]] || PARENT_REF="REPLACE_WITH_PARENT_RELATIVE_PATH"

[[ -f "${SOURCE_PATH}" ]] || die "source train script not found: ${SOURCE_PATH}"
[[ ! -e "${LEG_DIR}" ]] || die "leg already exists: ${LEG_DIR}"

mkdir -p "${LEG_DIR}"
cp -p "${SOURCE_PATH}" "${LEG_DIR}/train_gpt.py"

cat > "${LEG_DIR}/hypothesis.md" <<EOF
# Hypothesis — ${DATE_TAG}_${NAME_SLUG}

Parent: ${PARENT_REF}

## One Variable
- Name: \`REPLACE_ME\`
- New value: \`REPLACE_ME\`
- Baseline value: \`REPLACE_ME\`

## Why
- Explain why this one change should improve BPB under 600s wallclock.

## Gate Pass Criteria
- 1xGPU 2000-step gate improves versus control proxy.
- If no clear improvement: stop and mark DOES NOT PROMOTE.
EOF

cat > "${LEG_DIR}/ablation.md" <<'EOF'
# Ablation Log

## Gate (1xGPU, 2000 steps)
- Command:
- Seed:
- Best proxy metric:
- Verdict: PASS / FAIL
- Notes:

## Full Run (8xH100, 600s, seed=444)
- Command:
- final_sliding_window_exact val_bpb:
- Delta vs leader:
- Verdict: PROMOTION_CANDIDATE / FAIL

## Confirmation (8xH100, 600s, seed=300)
- final_sliding_window_exact val_bpb:
- Verdict: CONFIRMED / NOT_CONFIRMED
EOF

cat > "${LEG_DIR}/RESULTS.md" <<'EOF'
# RESULTS

**PENDING**

## Summary
- Hypothesis:
- Result:
- Delta vs leader:

## Evidence
- seed 444:
- seed 300:
- artifact size:

## Verdict
- PROMOTES / DOES NOT PROMOTE
- What to carry forward:
- What to avoid next:
EOF

cat > "${LEG_DIR}/tracked_env.sh" <<'EOF'
#!/usr/bin/env bash
# Edit this file, not the shell command line, when changing an experiment.
set -euo pipefail

export COMPRESSOR=brotli
export NUM_LAYERS=12
export QUANT_ATTN_BITS=5
export QUANT_MLP_BITS=6
export QUANT_AUX_BITS=6
export QUANT_EMBED_BITS=8
export QUANT_OTHER_BITS=8
export LOADER_MODE=coprime
export COPRIME_MAX_LOADED_SHARDS=1
export COPRIME_SHARDS_PER_BATCH=1
export COPRIME_SHARD_HOLD_STEPS=64
export COMPLEMENT_ALPHA=0
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2048
export ROPE_DIMS=16
export SWA_EVERY=50
export MTP_NUM_HEADS=0
export TRIGRAM=0
export NGRAM_EVAL_ORDER=0
export CUBRIC_CADENCE=0
export NGRAM_ENTROPY_SHIFT=0

# ONE VARIABLE RULE:
# Add the experiment change here as a tracked export.
# Example:
# export QK_GAIN_INIT=5.25
EOF

cat > "${LEG_DIR}/gate.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${REPO_ROOT}/__LEG_DIR__/train_gpt.py"
TRACKED_ENV="${REPO_ROOT}/__LEG_DIR__/tracked_env.sh"
LOG_DIR="${REPO_ROOT}/__LEG_DIR__/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gate_seed${SEED:-444}_$(date +%Y%m%d_%H%M%S).log"

NPROC="${NPROC_PER_NODE:-1}"
SEED="${SEED:-444}"

die() {
  echo "FATAL: $*" >&2
  exit 1
}

reject_adhoc_env() {
  local vars=(
    ITERATIONS WARMDOWN_ITERS MAX_WALLCLOCK_SECONDS SKIP_GPTQ
    COMPRESSOR NUM_LAYERS QUANT_ATTN_BITS QUANT_MLP_BITS QUANT_AUX_BITS
    QUANT_EMBED_BITS QUANT_OTHER_BITS LOADER_MODE COPRIME_MAX_LOADED_SHARDS
    COPRIME_SHARDS_PER_BATCH COPRIME_SHARD_HOLD_STEPS COMPLEMENT_ALPHA
    XSA_LAST_N BIGRAM_VOCAB_SIZE ROPE_DIMS SWA_EVERY MTP_NUM_HEADS
    TRIGRAM NGRAM_EVAL_ORDER CUBRIC_CADENCE NGRAM_ENTROPY_SHIFT
    VAL_LOSS_EVERY VOCAB_SIZE DATA_PATH TOKENIZER_PATH VE_ENABLED
    NUM_LOOPS QK_GAIN_INIT MATRIX_LR MUON_WD
  )
  local name=""
  for name in "${vars[@]}"; do
    if [[ -n "${!name+x}" ]]; then
      die "refusing ad-hoc env override: ${name} is already set in the shell. Edit ${TRACKED_ENV} instead."
    fi
  done
}

# Mandatory preflight: trainer diff must pass before the gate runs.
# For intentional framework-level legs, pass explicit overrides, e.g.:
# LEG_DIFF_GUARD_ARGS="--max-code-changes 80 --max-total-changed-lines 120" bash ${REPO_ROOT}/__LEG_DIR__/gate.sh
if [[ -n "${LEG_DIFF_GUARD_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python3 "${REPO_ROOT}/scripts/leg_diff_guard.py" "${REPO_ROOT}/__LEG_DIR__" ${LEG_DIFF_GUARD_ARGS}
else
  python3 "${REPO_ROOT}/scripts/leg_diff_guard.py" "${REPO_ROOT}/__LEG_DIR__"
fi

reject_adhoc_env
# shellcheck disable=SC1090
source "${TRACKED_ENV}"

SEED="${SEED}" \
NPROC_PER_NODE="${NPROC}" \
ITERATIONS=2000 \
WARMDOWN_ITERS=500 \
MAX_WALLCLOCK_SECONDS=4200 \
SKIP_GPTQ=1 \
VAL_LOSS_EVERY=500 \
torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
2>&1 | tee "${LOG_FILE}"

echo "LOG: ${LOG_FILE}"
EOF

cat > "${LEG_DIR}/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_SCRIPT="${REPO_ROOT}/__LEG_DIR__/train_gpt.py"
TRACKED_ENV="${REPO_ROOT}/__LEG_DIR__/tracked_env.sh"
LOG_DIR="${REPO_ROOT}/__LEG_DIR__/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/full_seed${SEED:-444}_$(date +%Y%m%d_%H%M%S).log"

NPROC="${NPROC_PER_NODE:-8}"
SEED="${SEED:-444}"

die() {
  echo "FATAL: $*" >&2
  exit 1
}

reject_adhoc_env() {
  local vars=(
    MAX_WALLCLOCK_SECONDS SKIP_GPTQ
    COMPRESSOR NUM_LAYERS QUANT_ATTN_BITS QUANT_MLP_BITS QUANT_AUX_BITS
    QUANT_EMBED_BITS QUANT_OTHER_BITS LOADER_MODE COPRIME_MAX_LOADED_SHARDS
    COPRIME_SHARDS_PER_BATCH COPRIME_SHARD_HOLD_STEPS COMPLEMENT_ALPHA
    XSA_LAST_N BIGRAM_VOCAB_SIZE ROPE_DIMS SWA_EVERY MTP_NUM_HEADS
    TRIGRAM NGRAM_EVAL_ORDER CUBRIC_CADENCE NGRAM_ENTROPY_SHIFT
    VOCAB_SIZE DATA_PATH TOKENIZER_PATH VE_ENABLED NUM_LOOPS
    QK_GAIN_INIT MATRIX_LR MUON_WD
  )
  local name=""
  for name in "${vars[@]}"; do
    if [[ -n "${!name+x}" ]]; then
      die "refusing ad-hoc env override: ${name} is already set in the shell. Edit ${TRACKED_ENV} instead."
    fi
  done
}

# Mandatory preflight: trainer diff must pass before the full run starts.
# For intentional framework-level legs, pass explicit overrides, e.g.:
# LEG_DIFF_GUARD_ARGS="--max-code-changes 80 --max-total-changed-lines 120" bash ${REPO_ROOT}/__LEG_DIR__/run.sh
if [[ -n "${LEG_DIFF_GUARD_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python3 "${REPO_ROOT}/scripts/leg_diff_guard.py" "${REPO_ROOT}/__LEG_DIR__" ${LEG_DIFF_GUARD_ARGS}
else
  python3 "${REPO_ROOT}/scripts/leg_diff_guard.py" "${REPO_ROOT}/__LEG_DIR__"
fi

reject_adhoc_env
# shellcheck disable=SC1090
source "${TRACKED_ENV}"

SEED="${SEED}" \
NPROC_PER_NODE="${NPROC}" \
MAX_WALLCLOCK_SECONDS=600 \
SKIP_GPTQ=1 \
torchrun --standalone --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
2>&1 | tee "${LOG_FILE}"

echo "LOG: ${LOG_FILE}"
EOF

sed -i "s#__LEG_DIR__#${LEG_DIR}#g" "${LEG_DIR}/gate.sh" "${LEG_DIR}/run.sh"

cat > "${LEG_DIR}/agent_allowlist.txt" <<EOF
# One path or prefix per line. Prefixes end with /
# scripts/agent_guard.sh check will fail any changed file not listed here.

${LEG_DIR}/
PIPELINE.md
EOF

cat > "${LEG_DIR}/AGENT_BRIEF.md" <<EOF
# Agent Brief — ${DATE_TAG}_${NAME_SLUG}

## Scope
You may edit only files listed in:
\`${LEG_DIR}/agent_allowlist.txt\`

## Task rules
1. Change exactly ONE variable vs baseline.
2. Do not edit \`vault/\`, \`records/\`, or \`LOCKED_SOTA/\`.
3. Update \`hypothesis.md\` before code edits.
4. Run \`python3 scripts/leg_diff_guard.py ${LEG_DIR}\` before any gate, full run, or commit.
5. Treat diff guard FAIL as a blocker unless the user explicitly approved wider thresholds.
6. Fill \`ablation.md\` and \`RESULTS.md\` after each run.

## Commands
\`\`\`bash
bash scripts/agent_guard.sh snapshot pre_${NAME_SLUG}
python3 scripts/leg_diff_guard.py ${LEG_DIR}
bash ${LEG_DIR}/gate.sh
bash ${LEG_DIR}/run.sh
bash scripts/agent_guard.sh check ${LEG_DIR}/agent_allowlist.txt --delta-latest
\`\`\`
EOF

chmod +x "${LEG_DIR}/gate.sh" "${LEG_DIR}/run.sh"
chmod +x "${LEG_DIR}/tracked_env.sh"

echo "Created: ${LEG_DIR}"
echo "Source copied to: ${LEG_DIR}/train_gpt.py"
echo "Next:"
echo "  1) Edit ${LEG_DIR}/hypothesis.md"
echo "  2) Edit ${LEG_DIR}/tracked_env.sh or ${LEG_DIR}/train_gpt.py"
echo "  3) Run: python3 scripts/leg_diff_guard.py ${LEG_DIR}"
echo "  4) Run: bash ${LEG_DIR}/gate.sh"
echo "  5) Verify scope: bash scripts/agent_guard.sh check ${LEG_DIR}/agent_allowlist.txt --delta-latest"
