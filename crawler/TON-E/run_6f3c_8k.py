#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


RUN_SCRIPT = Path(__file__).resolve()
ROOT = RUN_SCRIPT.parent
TRAIN_SCRIPT = ROOT / "train_gpt.py"
DEFAULT_DATASET_CANDIDATES = (
    Path("/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp8192"),
    ROOT.parent.parent / "data" / "datasets" / "fineweb10B_sp8192",
)
DEFAULT_TOKENIZER_CANDIDATES = (
    Path("/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_8192_bpe.model"),
    ROOT.parent.parent / "data" / "tokenizers" / "fineweb_8192_bpe.model",
)


@dataclass(frozen=True)
class RunConfig:
    seed: int
    world_size: int
    max_wallclock_seconds: int
    run_id: str
    data_path: Path
    tokenizer_path: Path

    def env(self) -> dict[str, str]:
        return {
            # Paths and identity.
            "RUN_ID": self.run_id,
            "SEED": str(self.seed),
            "DATA_PATH": str(self.data_path),
            "TOKENIZER_PATH": str(self.tokenizer_path),
            "MAX_WALLCLOCK_SECONDS": str(self.max_wallclock_seconds),
            # Lock out profile wrappers and pin the intended 8k crawler shape.
            "TON_E_RHYTHM": "0",
            "VOCAB_SIZE": "8192",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "USE_CRAWLER": "1",
            "NUM_FLAT_LAYERS": "6",
            "NUM_CRAWLER_LAYERS": "3",
            "CRAWLER_LOOPS": "3",
            # Keep the proven Nightcrawler-style block behavior explicit.
            "TIE_EMBEDDINGS": "1",
            "MLP_ACT": "relu_sq",
            "MLP_LEAKY_SLOPE": "0.5",
            "CRAWLER_MLP_LEAKY_SLOPE": "0.5",
            "ROPE_DIMS": "16",
            "VE_ENABLED": "1",
            "VE_LAYERS": "9,10",
            "XSA_LAST_N": "11",
            "XSA_INCLUDE_FLAT": "0",
            "CRAWLER_LOOP_ROPE_SCALES": "9,1,1",
            # Stage crawler compute in the same 33/66 schedule as the plain 8k base.
            "CRAWLER_COMPUTE_STAGED": "1",
            "CRAWLER_SAFE_WARMUP_STEPS": "0",
            "CRAWLER_GRADUATED": "1",
            "CRAWLER_START_FRAC": "0.0",
            "CRAWLER_RAMP_DELAY_FRAC": "0.33",
            "CRAWLER_RAMP_FRAC": "0.33",
            "CRAWLER_FORWARD_GRADUATED": "1",
            "CRAWLER_FORWARD_START_FRAC": "0.0",
            "CRAWLER_FORWARD_RAMP_DELAY_FRAC": "0.33",
            "CRAWLER_FORWARD_RAMP_FRAC": "0.33",
            # Proven optimization/export choices for the crawler line.
            "TIED_EMBED_LR": "0.035",
            "MATRIX_LR": "0.03",
            "EXPORT_QUANT": "int6",
            "SKIP_EMA": "1",
            "SKIP_GPTQ": "0",
            "LOOP_AWARE_GPTQ": "1",
            "GPTQ_CAL_SAMPLES": "256",
            "CRAWLER_QUANT_INT8": "0",
            "SIZE_TARGET_BYTES": "16000000",
            "SELECTIVE_PRUNE_ENABLE": "1",
            "SELECTIVE_PRUNE_FACTOR": "8",
            "COMPILE_ENABLED": "1",
            "COMPILE_FULLGRAPH": "0",
        }

    def summary(self) -> str:
        return "\n".join(
            [
                "Run configuration",
                f"  run_id: {self.run_id}",
                f"  world_size: {self.world_size}",
                f"  seed: {self.seed}",
                f"  wallclock_seconds: {self.max_wallclock_seconds}",
                f"  data_path: {self.data_path}",
                f"  tokenizer_path: {self.tokenizer_path}",
                "  topology: 6F+3C x3",
                "  vocab_size: 8192",
                "  model_dim: 512",
                "  export_quant: int6",
                "  gptq: loop-aware",
                "  crawler_schedule: staged 33/66",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical 8k 6F+3C Nightcrawler launcher."
    )
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-wallclock-seconds", type=int, default=600)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_existing_path(user_value: str, candidates: tuple[Path, ...], label: str) -> Path:
    if user_value:
        path = Path(user_value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.exists():
            return path
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"{label} not found. Searched:\n{searched}")


def build_config(args: argparse.Namespace) -> RunConfig:
    if args.world_size < 1:
        raise ValueError(f"--world-size must be >= 1, got {args.world_size}")
    if args.max_wallclock_seconds < 0:
        raise ValueError(
            f"--max-wallclock-seconds must be >= 0, got {args.max_wallclock_seconds}"
        )
    data_path = resolve_existing_path(
        args.data_path,
        DEFAULT_DATASET_CANDIDATES,
        "8k dataset",
    )
    tokenizer_path = resolve_existing_path(
        args.tokenizer_path,
        DEFAULT_TOKENIZER_CANDIDATES,
        "8k tokenizer",
    )
    run_id = args.run_id or f"nc3_8k_6f3c_s{args.seed}"
    return RunConfig(
        seed=args.seed,
        world_size=args.world_size,
        max_wallclock_seconds=args.max_wallclock_seconds,
        run_id=run_id,
        data_path=data_path,
        tokenizer_path=tokenizer_path,
    )


def build_command(config: RunConfig) -> list[str]:
    if config.world_size == 1:
        return [sys.executable, str(TRAIN_SCRIPT)]
    torchrun = shutil.which("torchrun")
    if not torchrun:
        raise FileNotFoundError("torchrun not found in PATH")
    return [torchrun, "--standalone", f"--nproc_per_node={config.world_size}", str(TRAIN_SCRIPT)]


def archive_outputs(config: RunConfig, started_at: float | None = None) -> None:
    artifact_dir = ROOT / "artifacts" / config.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    always_copy = (
        RUN_SCRIPT,
        TRAIN_SCRIPT,
        ROOT / "logs" / f"{config.run_id}.txt",
    )
    maybe_copy = (
        ROOT / "final_model.pt",
        ROOT / "final_model.int6.ptz",
        ROOT / "final_model.int8.ptz",
    )
    for source in always_copy:
        if source.exists():
            shutil.copy2(source, artifact_dir / source.name)
    for source in maybe_copy:
        if not source.exists():
            continue
        if started_at is not None and source.stat().st_mtime < started_at:
            continue
        shutil.copy2(source, artifact_dir / source.name)


def main() -> int:
    args = parse_args()
    config = build_config(args)
    command = build_command(config)
    print(config.summary())
    print(f"  command: {' '.join(command)}")
    if args.dry_run:
        return 0

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)

    env = os.environ.copy()
    env.update(config.env())

    started_at = time.time()
    try:
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    finally:
        archive_outputs(config, started_at=started_at)

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
