"""Run the same supervised benchmark as the UI and save a reproducible audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

from backend.foundation_benchmarks import PREDICTORS, run_benchmark_suite
from backend.predictions import PredictionBenchmarkConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="titeseq_4420_binding")
    parser.add_argument("--representation", choices=["ablang2", "sequence"], default="ablang2")
    parser.add_argument(
        "--predictors",
        choices=PREDICTORS,
        nargs="+",
        default=["ensemble"],
        help="Compare the ensemble and/or standalone frozen-embedding ridge predictors",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use cached ESM-2/AntiBERTy checkpoints; never download those weights",
    )
    parser.add_argument("--budget", type=int, default=768)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--split", choices=["random", "mutation_extrapolation"], default="random")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    config = PredictionBenchmarkConfig(
        task_id=args.task,
        representation=args.representation,
        train_limit=args.budget,
        repeats=args.seeds,
        seed=args.seed,
        test_limit=args.test_limit,
        split=args.split,
        predictors=tuple(args.predictors),
        embedding_batch_size=args.embedding_batch_size,
        local_only=args.local_only,
    )
    result = run_benchmark_suite(
        config,
        root / "runtime",
        tuple(args.predictors),
        lambda message: print(message, flush=True),
        args.embedding_batch_size,
        args.local_only,
    )
    packages = ["numpy", "torch", "scipy", "scikit-learn"]
    if "ensemble" in args.predictors and args.representation == "ablang2":
        packages.append("ablang2")
    if any(p in args.predictors for p in ("esm2-only", "antiberty-only")):
        packages += ["transformers", "safetensors", "huggingface-hub"]
    if "antiberty-only" in args.predictors:
        packages.append("antiberty")
    result["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "code_sha256": {
            name: hashlib.sha256((root / "backend" / name).read_bytes()).hexdigest()
            for name in (
                "prediction_models.py",
                "predictions.py",
                "public_data.py",
                "features.py",
                "foundation_benchmarks.py",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["curves"][-1], indent=2), flush=True)


if __name__ == "__main__":
    main()
