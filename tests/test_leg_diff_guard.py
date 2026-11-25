import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts" / "leg_diff_guard.py"

PARENT = """class Hyperparameters:
    foo = int(os.environ.get("FOO", "1"))
    bar = 2

def train():
    return 1
"""

CHILD_SINGLE_QUOTE = """class Hyperparameters:
    foo = int(os.environ.get('FOO', "2"))
    bar = 2

def train():
    return 1
"""

CHILD_CODE = """class Hyperparameters:
    foo = int(os.environ.get("FOO", "1"))
    bar = 2

def train():
    return 1

def extra():
    return 2
"""

CHILD_BRACKET = """class Hyperparameters:
    foo = int(os.environ.get("FOO", "1"))
    bar = int(os.environ["BAR"])

def train():
    return 1
"""

PARENT_TWO = """class Hyperparameters:
    foo = int(os.environ.get("FOO", "1"))
    bar = int(os.environ.get("BAR", "1"))

def train():
    return 1
"""


def run_guard(target: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GUARD), str(target), *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_single_quoted_env_counts_as_one_env_change(tmp_path: Path) -> None:
    parent = tmp_path / "parent.py"
    child = tmp_path / "child.py"
    parent.write_text(PARENT)
    child.write_text(CHILD_SINGLE_QUOTE)
    result = run_guard(child, "--parent", str(parent), "--no-diff")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hyperparam_envs_changed: FOO (limit 1)" in result.stdout
    assert "PASS:" in result.stdout
    assert "diff: suppressed (--no-diff)" in result.stdout


def test_code_change_outside_hyperparameters_fails(tmp_path: Path) -> None:
    parent = tmp_path / "parent.py"
    child = tmp_path / "child.py"
    parent.write_text(PARENT)
    child.write_text(CHILD_CODE)
    result = run_guard(child, "--parent", str(parent), "--no-diff")
    assert result.returncode == 1
    assert "FAIL:" in result.stdout
    assert "non-Hyperparameters" in result.stdout


def test_bracket_env_is_counted(tmp_path: Path) -> None:
    parent = tmp_path / "parent.py"
    child = tmp_path / "child.py"
    parent.write_text(PARENT_TWO)
    child.write_text(CHILD_BRACKET)
    result = run_guard(child, "--parent", str(parent), "--no-diff")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hyperparam_envs_changed: BAR (limit 1)" in result.stdout

    child.write_text(CHILD_BRACKET.replace('["BAR"]', '["BAZ"]'))
    too_many = run_guard(child, "--parent", str(parent), "--no-diff")
    assert too_many.returncode == 1
    assert "BAR" in too_many.stdout and "BAZ" in too_many.stdout
    assert "hyperparameter env vars" in too_many.stdout


def test_lock_tracks_parent_hash(tmp_path: Path) -> None:
    parent = tmp_path / "parent.py"
    child = tmp_path / "child.py"
    parent.write_text(PARENT)
    child.write_text(PARENT)
    lock = tmp_path / "lock.json"
    written = run_guard(child, "--parent", str(parent), "--no-diff", "--write-lock", "--lock-file", str(lock))
    assert written.returncode == 0, written.stdout + written.stderr
    payload = json.loads(lock.read_text())
    assert payload["target_sha256"]
    assert payload["parent_sha256"]

    checked = run_guard(child, "--parent", str(parent), "--check-lock", "--lock-file", str(lock), "--no-diff")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "lock OK:" in checked.stdout

    parent.write_text(PARENT + "\n")
    stale = run_guard(child, "--parent", str(parent), "--check-lock", "--lock-file", str(lock), "--no-diff")
    assert stale.returncode == 1
    assert "parent train_gpt.py hash differs" in stale.stdout

    lock.write_text("{not json")
    broken = run_guard(child, "--parent", str(parent), "--check-lock", "--lock-file", str(lock), "--no-diff")
    assert broken.returncode == 1
    assert "not valid JSON" in broken.stdout


def test_v2_5_gate_footprint_stays_inside_its_thresholds() -> None:
    result = run_guard(
        ROOT / "legs" / "2026-04-14_midnight_iii_v_2_5_online_gptq_tensor_seen",
        "--parent",
        str(ROOT / "legs" / "2026-04-14_midnight_iii_v_2_3_online_gptq_cpu" / "train_gpt.py"),
        "--max-code-changes",
        "300",
        "--max-total-changed-lines",
        "400",
        "--no-diff",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
