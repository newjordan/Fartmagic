import os
import subprocess
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)


def test_competition_dry_run_does_not_require_local_shards() -> None:
    env = os.environ.copy()
    env.pop("DATA_PATH", None)
    env.pop("TOKENIZER_PATH", None)
    env["DRY_RUN"] = "1"
    result = run(["bash", "scripts/run_competition.sh"], env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[TON-E dry-run]" in result.stdout
    assert "DATA_PATH=<unset>" in result.stdout
    assert "TOKENIZER_PATH=<unset>" in result.stdout

    env["DRY_RUN"] = "0"
    missing = run(["bash", "scripts/run_competition.sh"], env=env)
    assert missing.returncode == 1
    assert "Set DATA_PATH" in missing.stderr


def test_moonshot_dry_run_reports_paths() -> None:
    env = os.environ.copy()
    env.pop("DATA_PATH", None)
    env.pop("TOKENIZER_PATH", None)
    env["DRY_RUN"] = "1"
    result = run(["bash", "scripts/run_competition_moonshot.sh"], env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[MOONSHOT dry-run]" in result.stdout
    assert "DATA_PATH=<unset>" in result.stdout


def test_lane_help_names_the_clean_campaigns() -> None:
    iii = run(["bash", "scripts/run_midnight_iii_lane.sh", "--help"])
    assert iii.returncode == 0, iii.stdout + iii.stderr
    assert "2026-04-12_midnight_iii_clean" in iii.stdout
    twelve = run(["bash", "scripts/run_midnight_12l_lane.sh", "--help"])
    assert twelve.returncode == 0, twelve.stdout + twelve.stderr
    assert "2026-04-12_midnight_12l_clean" in twelve.stdout
    bad = run(["bash", "scripts/run_midnight_iii_lane.sh", "train"])
    assert bad.returncode == 1
    assert "gate or run" in bad.stderr


def test_launch_guard_falls_back_when_torchrun_is_absent(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python3 = bindir / "python3"
    python3.write_text("#!/bin/sh\nexit 0\n")
    python3.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bindir}:/usr/bin:/bin"
    script = """
set -euo pipefail
source scripts/launch_guard.sh
lg_ensure_torchrun
type torchrun
"""
    result = run(["bash", "-c", script], env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "launcher=python3 -m torch.distributed.run" in result.stdout
    assert "function" in result.stdout

    python3.write_text("#!/bin/sh\nexit 1\n")
    failed = run(["bash", "-c", script], env=env)
    assert failed.returncode == 1
    assert "torchrun not found" in failed.stderr


def test_launch_guard_rejects_a_world_size_above_the_visible_gpus() -> None:
    ok = run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
source scripts/launch_guard.sh
lg_visible_gpus() { echo 1; }
lg_assert_world_size 1
""",
        ]
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "visible_gpus=1" in ok.stdout

    blocked = run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
source scripts/launch_guard.sh
lg_visible_gpus() { echo 1; }
lg_assert_world_size 2
""",
        ]
    )
    assert blocked.returncode == 1
    assert "world-size mismatch" in blocked.stderr


def test_agent_guard_allowlist_and_quoted_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    run(["git", "init"], cwd=repo)
    run(["git", "config", "user.email", "guard@example.com"], cwd=repo)
    run(["git", "config", "user.name", "guard"], cwd=repo)
    (repo / "ok.txt").write_text("ok\n")
    run(["git", "add", "ok.txt"], cwd=repo)
    run(["git", "commit", "-m", "init"], cwd=repo)
    (repo / "keep.txt").write_text("keep\n")
    (repo / "foo bar.txt").write_text("space\n")
    (repo / "nope.txt").write_text("nope\n")
    (repo / "allow.txt").write_text("keep.txt\nfoo bar.txt\n")

    status = run(["bash", str(ROOT / "scripts" / "agent_guard.sh"), "status"], cwd=repo)
    assert status.returncode == 0, status.stdout + status.stderr
    assert "foo bar.txt" in status.stdout
    assert '"foo bar.txt"' not in status.stdout

    checked = run(["bash", str(ROOT / "scripts" / "agent_guard.sh"), "check", "allow.txt"], cwd=repo)
    assert checked.returncode == 2
    assert "nope.txt" in checked.stdout
    assert "keep.txt" not in checked.stdout.split("FAIL:", 1)[-1]


def test_new_leg_points_at_itself_and_matches_its_parent() -> None:
    slug = f"zz_probe_{uuid.uuid4().hex[:8]}"
    created = ""
    try:
        result = run(["bash", "scripts/new_leg.sh", slug])
        assert result.returncode == 0, result.stdout + result.stderr
        created_line = next(line for line in result.stdout.splitlines() if line.startswith("created: "))
        created = created_line.split(": ", 1)[1]
        leg = ROOT / created
        gate = (leg / "gate.sh").read_text()
        assert 'dirname -- "${BASH_SOURCE[0]}"' in gate
        assert "launch_guard.sh" in gate
        assert (leg / "agent_allowlist.txt").read_text().count(created + "/") == 1
        guard = run(["python3", "scripts/leg_diff_guard.py", created, "--no-diff"])
        assert guard.returncode == 0, guard.stdout + guard.stderr
    finally:
        if created:
            target = ROOT / created
            if target.exists():
                import shutil
                shutil.rmtree(target)


def test_brand_banner_prints_when_enabled() -> None:
    leg = ROOT / "legs" / "2026-04-14_midnight_iii_v_2_4_brand_bookend"
    gate = (leg / "gate.sh").read_text()
    start = gate.index("print_banner()")
    end = gate.index('print_banner "pre"')
    banner = leg / "brand_banner.braille.txt"
    script = gate[start:end] + '\nprint_banner pre\n'
    shown = run(
        ["bash", "-c", script],
        env={**os.environ, "RUN_BRAND_BANNER": "1", "BANNER_FILE": str(banner)},
    )
    assert shown.returncode == 0, shown.stdout + shown.stderr
    assert "brand_banner:pre:start" in shown.stdout
    assert "MIDNIGHT" in shown.stdout
    assert "brand_banner:pre:end" in shown.stdout

    missing = run(
        ["bash", "-c", script],
        env={**os.environ, "RUN_BRAND_BANNER": "1", "BANNER_FILE": str(leg / "missing-banner.txt")},
    )
    assert missing.returncode == 1
    assert "banner file is missing" in missing.stderr
