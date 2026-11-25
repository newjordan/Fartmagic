"""Download challenge FineWeb shards and tokenizers from Hugging Face.

Model checkpoints are out of scope. This script materializes dataset bins
and SentencePiece tokenizer files declared by the dataset manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
TOKENIZERS_DIR = ROOT / "tokenizers"


def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def local_path_for_remote(relative_path: str) -> Path:
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return DATASETS_DIR.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
    return ROOT / remote_path


def prepare_destination(destination: Path) -> str:
    """Return 'skip' when a real file is already present, else 'fetch'.

    Symlinks are removed so a later cache eviction cannot leave a broken link
    in data/. A directory at the destination is an error, not a shard.
    """
    if destination.is_symlink():
        destination.unlink()
    if destination.is_dir():
        raise ValueError(f"destination is a directory, expected a file: {destination}")
    if destination.is_file():
        return "skip"
    return "fetch"


def _hf_download(relative_path: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required to download shards. "
            "Install it with: python3 -m pip install -e '.[data]'"
        ) from exc

    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    return cached_path.resolve(strict=True)


def get(relative_path: str) -> str:
    destination = local_path_for_remote(relative_path)
    action = prepare_destination(destination)
    if action == "skip":
        print(f"fineweb: skip {relative_path}", file=sys.stderr)
        return "skip"

    cached_source = _hf_download(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)
    print(f"fineweb: fetch {relative_path}", file=sys.stderr)
    return "fetch"


def manifest_path() -> Path:
    return local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/manifest.json")


def load_manifest(*, skip_manifest_download: bool) -> dict:
    path = manifest_path()
    if not path.is_file():
        if skip_manifest_download:
            raise FileNotFoundError(
                f"manifest.json is required for manifest-driven shard counts but is not present locally at {path}"
            )
        get(f"{REMOTE_ROOT_PREFIX}/manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_paths_for_tokenizer(tokenizer_entry: dict) -> list[str]:
    artifacts = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            artifacts.append(str(value))
    if not artifacts:
        raise ValueError(f"tokenizer entry is missing downloadable artifacts: {tokenizer_entry}")
    return artifacts


def require_int_stat(dataset_entry: dict, key: str, dataset_dir: str) -> int:
    stats = dataset_entry.get("stats") or {}
    raw = stats.get(key)
    if raw is None:
        raise ValueError(f"dataset {dataset_dir} is missing stats.{key} in the manifest")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dataset {dataset_dir} stats.{key} is not an integer: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"dataset {dataset_dir} stats.{key} must be >= 0, got {value}")
    return value


def resolve_train_shards(positional: int | None, flagged: int, flagged_default: int) -> int:
    if positional is None:
        train_shards = flagged
    elif flagged != flagged_default and flagged != positional:
        raise ValueError(
            "pass the training shard count either as a positional argument or as --train-shards, not both "
            f"(positional={positional}, --train-shards={flagged})"
        )
    else:
        train_shards = positional
    if train_shards < 0:
        raise ValueError("train_shards must be non-negative")
    return train_shards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download challenge FineWeb shards and tokenizers from Hugging Face. Does not download model weights."
    )
    parser.add_argument(
        "train_shards_positional",
        nargs="?",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=80,
        help="Number of training shards to download for the selected variant. Defaults to 80.",
    )
    parser.add_argument(
        "--variant",
        default="sp1024",
        help="Tokenizer family to download, for example sp1024, sp4096, or byte260.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip downloading manifest.json and require a local copy.",
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Also download docs_selected.jsonl and its sidecar for tokenizer retraining or dataset re-export.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataset_dir = dataset_dir_for_variant(args.variant)
        train_shards = resolve_train_shards(
            args.train_shards_positional,
            args.train_shards,
            parser.get_default("train_shards"),
        )
        manifest = load_manifest(skip_manifest_download=args.skip_manifest)
        dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
        if dataset_entry is None:
            raise ValueError(f"dataset {dataset_dir} not found in {REMOTE_ROOT_PREFIX}/manifest.json")
        max_train_shards = require_int_stat(dataset_entry, "files_train", dataset_dir)
        val_shards = require_int_stat(dataset_entry, "files_val", dataset_dir)
        if train_shards > max_train_shards:
            raise ValueError(
                f"{args.variant} only has {max_train_shards} training shards on {REPO_ID}, requested {train_shards}"
            )
        tokenizer_name = dataset_entry.get("tokenizer_name")
        tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_entry is None:
            raise ValueError(f"tokenizer {tokenizer_name} not found in {REMOTE_ROOT_PREFIX}/manifest.json")

        fetched = 0
        skipped = 0

        def tally(action: str) -> None:
            nonlocal fetched, skipped
            if action == "fetch":
                fetched += 1
            else:
                skipped += 1

        if args.with_docs:
            tally(get(f"{REMOTE_ROOT_PREFIX}/docs_selected.jsonl"))
            tally(get(f"{REMOTE_ROOT_PREFIX}/docs_selected.source_manifest.json"))

        dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{dataset_dir}"
        for i in range(val_shards):
            tally(get(f"{dataset_prefix}/fineweb_val_{i:06d}.bin"))
        for i in range(train_shards):
            tally(get(f"{dataset_prefix}/fineweb_train_{i:06d}.bin"))
        for artifact_path in artifact_paths_for_tokenizer(tokenizer_entry):
            tally(get(f"{REMOTE_ROOT_PREFIX}/{artifact_path}"))
    except (ValueError, FileNotFoundError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        f"fineweb: done variant={args.variant} train_shards={train_shards} "
        f"val_shards={val_shards} fetched={fetched} skipped={skipped}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
