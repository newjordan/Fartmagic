import compileall
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOME_MARKER = "/home/" + "frosty40"


def test_python_sources_parse() -> None:
    roots = ["scripts", "data", "vault", "legs", "gemini_runs", "tests"]
    skip = re.compile(r"(__pycache__|\.git)")
    for name in roots:
        assert compileall.compile_dir(str(ROOT / name), quiet=1, rx=skip), name


def test_pod_stack_lock_matches_the_setup_script() -> None:
    import subprocess

    result = subprocess.run(
        ["bash", "scripts/pod_stack_guard.sh", "verify"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_personal_home_paths_in_tracked_text() -> None:
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", "logs", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.suffix not in {".py", ".sh", ".md", ".txt", ".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if HOME_MARKER in text:
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_gates_launch_their_own_trainer() -> None:
    for gate in sorted((ROOT / "legs").glob("*/gate.sh")):
        text = gate.read_text()
        own = gate.parent.name
        train_lines = [line for line in text.splitlines() if line.startswith("TRAIN_SCRIPT=")]
        assert train_lines, gate
        for line in train_lines:
            assert own in line or "${LEG_DIR}" in line, f"{gate}: {line}"
        if "${LEG_DIR}" in text:
            assert own in text


def test_allowlists_and_briefs_name_their_own_leg() -> None:
    for allow in sorted((ROOT / "legs").glob("*/agent_allowlist.txt")):
        own = allow.parent.name
        assert f"legs/{own}/" in allow.read_text(), allow
    for brief in sorted((ROOT / "legs").glob("*/AGENT_BRIEF.md")):
        own = brief.parent.name
        assert own in brief.read_text().splitlines()[0], brief


def test_shell_entrypoints_parse() -> None:
    import subprocess

    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    scripts.extend(sorted((ROOT / "legs").glob("*/*.sh")))
    assert scripts
    result = subprocess.run(["bash", "-n", *map(str, scripts)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
