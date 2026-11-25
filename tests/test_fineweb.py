import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "data" / "cached_challenge_fineweb.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cached_challenge_fineweb", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variant_names_and_rejections() -> None:
    module = load_module()
    assert module.dataset_dir_for_variant("byte260") == "fineweb10B_byte260"
    assert module.dataset_dir_for_variant("sp8192") == "fineweb10B_sp8192"
    try:
        module.dataset_dir_for_variant("sp")
    except ValueError as exc:
        assert "unsupported variant" in str(exc)
    else:
        raise AssertionError("sp should be rejected")


def test_remote_paths_land_under_data_dirs() -> None:
    module = load_module()
    assert module.local_path_for_remote("datasets/manifest.json") == module.ROOT / "manifest.json"
    shard = module.local_path_for_remote("datasets/datasets/fineweb10B_sp1024/fineweb_train_000000.bin")
    assert shard == module.DATASETS_DIR / "fineweb10B_sp1024" / "fineweb_train_000000.bin"
    tok = module.local_path_for_remote("datasets/tokenizers/fineweb_1024_bpe.model")
    assert tok == module.TOKENIZERS_DIR / "fineweb_1024_bpe.model"


def test_train_shard_flags_conflict() -> None:
    module = load_module()
    assert module.resolve_train_shards(None, 10, 80) == 10
    assert module.resolve_train_shards(4, 80, 80) == 4
    assert module.resolve_train_shards(4, 4, 80) == 4
    try:
        module.resolve_train_shards(4, 9, 80)
    except ValueError as exc:
        assert "not both" in str(exc)
    else:
        raise AssertionError("conflicting shard counts should fail")
    try:
        module.resolve_train_shards(None, -1, 80)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative shard count should fail")


def test_missing_manifest_stat_is_a_clear_error() -> None:
    module = load_module()
    try:
        module.require_int_stat({"stats": {}}, "files_train", "fineweb10B_sp1024")
    except ValueError as exc:
        assert "stats.files_train" in str(exc)
    else:
        raise AssertionError("missing stat should fail")
    try:
        module.require_int_stat({"stats": {"files_val": "nope"}}, "files_val", "fineweb10B_sp1024")
    except ValueError as exc:
        assert "not an integer" in str(exc)
    else:
        raise AssertionError("non-integer stat should fail")


def test_prepare_destination_replaces_symlinks_and_keeps_files(tmp_path: Path) -> None:
    module = load_module()
    real = tmp_path / "real.bin"
    real.write_text("shard")
    link = tmp_path / "linked.bin"
    link.symlink_to(real)
    assert module.prepare_destination(link) == "fetch"
    assert not link.exists()

    kept = tmp_path / "kept.bin"
    kept.write_text("already")
    assert module.prepare_destination(kept) == "skip"
    assert kept.read_text() == "already"

    folder = tmp_path / "dir.bin"
    folder.mkdir()
    try:
        module.prepare_destination(folder)
    except ValueError as exc:
        assert "directory" in str(exc)
    else:
        raise AssertionError("directory destination should fail")


def test_help_and_bad_variant_do_not_download() -> None:
    help_run = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_run.returncode == 0
    assert "model weights" in help_run.stdout

    bad = subprocess.run(
        [sys.executable, str(SCRIPT), "--variant", "nope"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad.returncode == 2
    assert "FATAL:" in bad.stderr
    assert "unsupported variant" in bad.stderr
