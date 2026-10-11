"""Target/assay-isolated prediction jobs and leakage-controlled retrospective tests."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.stats import spearmanr

from backend.models import SequenceRecord
from backend.prediction_models import PredictionEnsemble, antibody_embeddings, residue_codes
from backend.public_data import fingerprint, load_task


class PredictionConfig(BaseModel):
    """Explicit model, compute and assay choices; no acquisition parameters."""

    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(default="titeseq_4420_binding", max_length=100)
    representation: Literal["ablang2", "sequence"] = "ablang2"
    train_limit: int = Field(
        default=1024, ge=0, le=12000, description="0 uses all available development rows"
    )
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    coverage: Literal[0.8, 0.9] = 0.9
    queries: list[SequenceRecord] | None = Field(default=None, min_length=1, max_length=2000)


class PredictionBenchmarkConfig(BaseModel):
    """A fixed hidden test set at increasing labeled-data budgets, never an AL loop."""

    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(default="titeseq_4420_binding", max_length=100)
    representation: Literal["ablang2", "sequence"] = "ablang2"
    predictors: tuple[Literal["ensemble", "esm2-only", "antiberty-only"], ...] = Field(
        default=("ensemble",), min_length=1, max_length=3
    )
    embedding_batch_size: int = Field(default=16, ge=1, le=128)
    local_only: bool = False
    train_limit: int = Field(default=768, ge=80, le=12000)
    test_limit: int = Field(default=512, ge=20, le=2000)
    repeats: int = Field(default=3, ge=1, le=10)
    seed: int = Field(default=42, ge=0, le=2**31 - 11)
    split: Literal["random", "mutation_extrapolation"] = "random"
    top_k: int = Field(default=24, ge=1, le=96)

    @field_validator("predictors")
    @classmethod
    def unique_predictors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate runs of the same predictor."""
        if len(set(value)) != len(value):
            raise ValueError("Choose each benchmark predictor only once")
        return value


def partition(records: list[dict], seed: int, split: str = "random") -> tuple[np.ndarray, np.ndarray]:
    """Partition unique sequence pairs without consulting scores."""
    if len({fingerprint(r) for r in records}) != len(records):
        raise ValueError("Duplicate sequence pairs must be aggregated before splitting")
    rng = np.random.default_rng(seed)
    if split == "random":
        order = rng.permutation(len(records))
        ntest = max(12, math.ceil(len(records) * 0.2))
        test, development = order[:ntest], order[ntest:]
    else:
        # This unsupervised split uses sequence identities only. Whole mutation-count
        # strata remain on one side of the boundary, including cutoff ties.
        codes = residue_codes(records)
        reference = np.array([np.bincount(c, minlength=21).argmax() for c in codes.T])
        distance = (codes != reference).sum(axis=1)
        threshold = float(np.quantile(distance, 0.8))
        test = np.flatnonzero(distance > threshold)
        development = rng.permutation(np.flatnonzero(distance <= threshold))
        if len(test) < 12:
            raise ValueError(
                "This assay has too few higher-mutation variants for the extrapolation split; use the random split"
            )
    if len(development) < 60:
        raise ValueError("This task is too small for independent development and test sets")
    return development, test


def metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    maximize: bool,
    top_k: int,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict:
    """Tie-inclusive top-5% recall and a fixed-budget enrichment against random."""
    direction = 1 if maximize else -1
    actual, scores = direction * truth, direction * predicted
    threshold = np.sort(actual)[-max(1, math.ceil(0.05 * len(actual)))]
    top = actual >= threshold
    k = min(top_k, len(actual))
    order = np.argsort(-scores, kind="stable")[:k]
    hits = int(top[order].sum())
    rho = (
        float(spearmanr(truth, predicted).statistic) if np.ptp(truth) > 0 and np.ptp(predicted) > 0 else None
    )
    result = dict(
        spearman=rho,
        rmse=float(np.sqrt(np.mean((truth - predicted) ** 2))),
        mae=float(np.mean(np.abs(truth - predicted))),
        top_recall=hits / int(top.sum()),
        top_enrichment=(hits / k) / float(top.mean()),
        top_hits=hits,
        top_population=int(top.sum()),
        top_k=k,
        test_count=len(truth),
        random_recall=k / len(truth),
        random_enrichment=1.0,
    )
    if lower is not None:
        result.update(
            coverage=float(np.mean((truth >= lower) & (truth <= upper))),
            mean_interval_width=float(np.mean(upper - lower)),
        )
    return result


def _embeddings(
    records: list[dict], mode: str, root: Path, progress: Callable[[str], None]
) -> np.ndarray | None:
    return antibody_embeddings(records, root / "ablang2-cache", progress) if mode == "ablang2" else None


def benchmark_plans(
    config: PredictionBenchmarkConfig, records: list[dict]
) -> list[tuple[int, np.ndarray, np.ndarray, list[int]]]:
    """Shared label-free outer splits and nested budgets for all benchmark predictors."""
    plans = []
    for repeat in range(config.repeats):
        seed = config.seed + repeat
        development, test = partition(records, seed, config.split)
        development = development[: config.train_limit]
        test = np.random.default_rng(seed + 12345).permutation(test)[: config.test_limit]
        sizes = sorted({max(60, int(len(development) * f)) for f in (0.25, 0.5, 1.0)})
        plans.append((seed, development, test, sizes))
    return plans


def run_prediction(
    config: PredictionConfig, root: Path, progress: Callable[[str], None] = lambda _: None
) -> dict:
    """Rank supplied variants, or demonstrate on unseen public assay sequences."""
    start = time.monotonic()
    metadata, records = load_task(config.task_id, root / "custom-tasks")
    if not metadata["ready"]:
        raise ValueError(
            "This task is catalogued for reference but has insufficient unique labeled sequences"
        )
    progress("Preparing target-specific training and query sequences")
    if config.queries is None:
        development, test = partition(records, config.seed)
        queries = [
            {k: records[i][k] for k in ("variant_id", "heavy_sequence", "light_sequence")} for i in test[:512]
        ]
        mode = "held_out_demo"
    else:
        development = np.random.default_rng(config.seed).permutation(len(records))
        queries = [r.model_dump() for r in config.queries]
        if len({r["variant_id"] for r in queries}) != len(queries):
            raise ValueError("Query variant IDs must be unique")
        mode = "uploaded_candidates"
    available = len(development)
    if config.train_limit:
        development = development[: config.train_limit]
    training = [records[i] for i in development]
    if len(training) < 60:
        raise ValueError("Training limit must leave at least 60 unique examples")
    encoded = _embeddings(training + queries, config.representation, root, progress)
    train_embedding = encoded[: len(training)] if encoded is not None else None
    query_embedding = encoded[len(training) :] if encoded is not None else None
    engine = PredictionEnsemble(config.seed, config.coverage).fit(
        training, np.array([r["score"] for r in training]), train_embedding, progress
    )
    progress("Predicting native assay scores and sequence support")
    predictions = engine.predict(queries, query_embedding)
    known = {fingerprint(r): r["score"] for r in records}
    included = {fingerprint(r) for r in training}
    # Support includes chain layout, not just a high heavy-chain positional match.
    layouts = {(bool(r["heavy_sequence"]), bool(r["light_sequence"])) for r in training}
    output = []
    for i, row in enumerate(queries):
        key = fingerprint(row)
        supported = (
            predictions["nearest_identity"][i] >= 0.9
            and (bool(row["heavy_sequence"]), bool(row["light_sequence"])) in layouts
        )
        output.append(
            {
                **row,
                "prediction": float(predictions["prediction"][i]),
                "lower": float(predictions["lower"][i]),
                "upper": float(predictions["upper"][i]),
                "model_disagreement": float(predictions["disagreement"][i]),
                "nearest_training_identity": float(predictions["nearest_identity"][i]),
                "support": "within_sequence_neighborhood"
                if supported
                else "outside_validated_sequence_neighborhood",
                "used_as_labeled_data": key in included,
                "observed_score": known.get(key),
            }
        )
    direction = 1 if metadata["maximize"] else -1
    output.sort(key=lambda r: (r["support"] != "within_sequence_neighborhood", -direction * r["prediction"]))
    for rank, row in enumerate(output, 1):
        row["rank"] = rank
    evaluation = None
    if mode == "held_out_demo":
        evaluation = metrics(
            np.array([known[fingerprint(r)] for r in queries]),
            predictions["prediction"],
            metadata["maximize"],
            24,
            predictions["lower"],
            predictions["upper"],
        )
    return dict(
        kind="prediction",
        task=metadata,
        config=config.model_dump(exclude={"queries"}),
        mode=mode,
        variants=output,
        model_card=engine.card(),
        labeled_available=available,
        labeled_used=len(training),
        evaluation=evaluation,
        seconds=time.monotonic() - start,
        warnings=[
            "Scores are specific to this antigen, scaffold family and assay; they do not establish clinical efficacy or universal antibody quality.",
            "Out-of-neighborhood sequences appear last and need target-specific experimental validation.",
            "Expression assays are not measurements of structural stability.",
            "Predictions on supplied sequences already in labeled data are marked and must not be used as benchmark evidence.",
        ],
    )


def run_prediction_benchmark(
    config: PredictionBenchmarkConfig, root: Path, progress: Callable[[str], None] = lambda _: None
) -> dict:
    """Repeated held-out learning curves; fit functions never receive oracle test labels."""
    started = time.monotonic()
    metadata, records = load_task(config.task_id, root / "custom-tasks")
    plans, union = benchmark_plans(config, records), set()
    for _, development, test, _ in plans:
        union.update(development.tolist())
        union.update(test.tolist())
    # Frozen embeddings can be shared across splits: no labels or fit statistics used.
    indices = sorted(union)
    encoded = _embeddings([records[i] for i in indices], config.representation, root, progress)
    lookup = {index: pos for pos, index in enumerate(indices)}

    def embedding(ids):
        return encoded[[lookup[int(i)] for i in ids]] if encoded is not None else None

    rounds, audits = [], []
    for seed, development, test, sizes in plans:
        test_records = [records[i] for i in test]
        truth = np.array([r["score"] for r in test_records])
        for size in sizes:
            chosen = development[:size]
            progress(f"Benchmark seed {seed}: {size} labeled sequences; {len(test)} hidden test sequences")
            train = [records[i] for i in chosen]
            engine = PredictionEnsemble(seed).fit(
                train, np.array([r["score"] for r in train]), embedding(chosen), progress
            )
            pred = engine.predict(test_records, embedding(test), support=False)
            item = dict(
                seed=seed,
                training_size=size,
                ensemble=metrics(
                    truth,
                    pred["prediction"],
                    metadata["maximize"],
                    config.top_k,
                    pred["lower"],
                    pred["upper"],
                ),
                additive=metrics(truth, pred["additive"], metadata["maximize"], config.top_k),
                mean=metrics(truth, pred["mean_baseline"], metadata["maximize"], config.top_k),
                model_card=engine.card(),
            )
            rounds.append(item)
        audits.append(
            dict(
                seed=seed,
                training_ids=[records[i]["variant_id"] for i in development],
                test_ids=[r["variant_id"] for r in test_records],
                test_hash=hashlib.sha256(
                    "|".join(sorted(fingerprint(r) for r in test_records)).encode()
                ).hexdigest(),
            )
        )
    curves = []
    for size in sorted({r["training_size"] for r in rounds}):
        values = [r for r in rounds if r["training_size"] == size]
        point = {"training_size": size, "repeats": len(values)}
        for model in ("ensemble", "additive", "mean"):
            point[model] = {}
            for key in values[0][model]:
                numbers = [r[model][key] for r in values if r[model][key] is not None]
                point[model][key] = float(np.mean(numbers)) if numbers else None
        point["recall_range"] = [
            min(r["ensemble"]["top_recall"] for r in values),
            max(r["ensemble"]["top_recall"] for r in values),
        ]
        curves.append(point)
    return dict(
        kind="prediction_benchmark",
        task=metadata,
        config=config.model_dump(),
        curves=curves,
        runs=rounds,
        split_audit=audits,
        seconds=time.monotonic() - started,
        protocol="Exact sequence pairs are deduplicated before splitting. Test identities are fixed across budgets within each seed. Hyperparameters and ensemble weights use internal validation only; intervals use separate calibration labels. No sequential acquisition or oracle feasibility filter.",
        limitations=[
            "Random library splits assess within-assay interpolation, not transfer to a new antigen.",
            "Mutation extrapolation holds out whole higher-distance strata relative to the unlabeled library consensus; it is not a held-out-scaffold test.",
            "Replicate medians discard replicate noise and censoring; censored affinity endpoints are treated as point measurements.",
            "Frozen foundation-model pretraining overlap cannot be excluded.",
            "Seed ranges are descriptive, not confidence intervals or independent biological replicates.",
        ],
    )
