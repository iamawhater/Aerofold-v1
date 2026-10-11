"""Public DMS closed-loop simulation with isolated policies and oracle-only evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from pydantic import Field

from backend.features import fragment_mutation_embeddings
from backend.learning import acquisition_scores, fit_objectives
from backend.models import StrictModel
from backend.optimization import hypervolume, select_diverse

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
DATASETS = {"titeseq_4_4_20": "titeseq_4_4_20"}
logger = logging.getLogger("uvicorn.error.aerofold.benchmarks")


class BenchmarkConfig(StrictModel):
    dataset_id: str = "titeseq_4_4_20"
    rounds: int = Field(default=5, ge=1, le=12)
    batch_size: Literal[12, 24, 48] = 24
    seed_size: int = Field(default=20, ge=6, le=100)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    exploration: float = Field(default=0.35, ge=0, le=1)
    diversity: float = Field(default=0.55, ge=0, le=1)
    min_stability: float = Field(default=20, ge=0, le=100)
    confidence_beta: float = Field(default=1, ge=0, le=3)


@dataclass(frozen=True)
class BenchmarkData:
    """Oracle arrays belong to the runner/evaluator; policy functions cannot see them."""

    metadata: dict
    ids: list[str]
    sequences: list[str]
    oracle: np.ndarray


def load_dataset(dataset_id: str) -> BenchmarkData:
    """Read only an allowlisted packaged CSV and verify its manifest checksum."""
    if dataset_id not in DATASETS:
        raise ValueError("Unknown benchmark dataset; use GET /api/benchmark/datasets")
    stem = DATASETS[dataset_id]
    metadata = json.loads((DATA_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    raw = (DATA_DIR / f"{stem}.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["csv_sha256"]:
        raise ValueError("Packaged benchmark checksum mismatch; restore the CSV and manifest together")
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    ids, sequences = [r["variant_id"] for r in rows], [r["variant_sequence"] for r in rows]
    oracle = np.array([[float(r["true_binding"]), float(r["stability_score"])] for r in rows])
    if (
        len(rows) != metadata["variant_count"]
        or len(set(ids)) != len(ids)
        or len(set(sequences)) != len(sequences)
    ):
        raise ValueError("Invalid benchmark cardinality or duplicate sequences/IDs")
    if not np.isfinite(oracle).all() or np.any((oracle[:, 1] < 0) | (oracle[:, 1] > 100)):
        raise ValueError("Invalid benchmark oracle scores")
    if any(r["parent_sequence"] != metadata["parent_sequence"] for r in rows):
        raise ValueError("Inconsistent benchmark parent")
    if any(len(s) != 20 or set(s) - set("ACDEFGHIKLMNPQRSTVWY") for s in sequences):
        raise ValueError("Invalid benchmark CDR fragment sequence")
    oracle.flags.writeable = False
    return BenchmarkData(metadata, ids, sequences, oracle)


def list_datasets() -> list[dict]:
    """Catalog contains provenance and score semantics, never hidden labels."""
    return [load_dataset(identifier).metadata for identifier in DATASETS]


def validate_budget(config: BenchmarkConfig, size: int) -> None:
    if config.seed_size + config.rounds * config.batch_size > size:
        raise ValueError(f"Budget exceeds {size} variants: seed_size + rounds × batch_size must be <= {size}")


def spearman(predictions: np.ndarray, truth: np.ndarray) -> float | None:
    """Pearson correlation of average tie ranks; undefined returns JSON null."""
    if len(truth) < 2:
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts + 1) / 2)[inverse]

    a, b = ranks(predictions), ranks(truth)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def fit_policy(
    embeddings: np.ndarray, observed: np.ndarray, labels: np.ndarray, config: BenchmarkConfig
) -> dict:
    """Receive ONLY revealed labels; derive feasibility without oracle knowledge."""
    means, sigmas, covariances = fit_objectives(embeddings, observed, labels)
    # pKd values are measured only within this assay's [5, 9.5] range.
    # A Gaussian extrapolation outside it is not a distinguishable oracle score.
    means[:, 0] = np.clip(means[:, 0], 5.0, 9.5)
    unseen = np.ones(len(embeddings), dtype=bool)
    unseen[observed] = False
    eligible = unseen & (means[:, 1] - config.confidence_beta * sigmas[:, 1] >= config.min_stability)
    return {
        "means": means,
        "sigmas": sigmas,
        "covariances": covariances,
        "unseen": unseen,
        "eligible": eligible,
    }


def choose_batch(
    policy: dict,
    embeddings: np.ndarray,
    observed_labels: np.ndarray,
    config: BenchmarkConfig,
    rng: np.random.Generator,
    agent: str,
    round_number: int,
) -> list[int]:
    """Either DPP/MC-EHVI or uniform sampling under the same learned gate."""
    candidates = np.flatnonzero(policy["eligible"])
    if len(candidates) < config.batch_size:
        raise ValueError(
            f"{agent} has only {len(candidates)} feasible candidates; {config.batch_size} are required"
        )
    if agent == "random":
        return rng.choice(candidates, config.batch_size, replace=False).tolist()
    scores, _, _ = acquisition_scores(
        policy["means"],
        policy["sigmas"],
        observed_labels,
        policy["unseen"],
        config.exploration,
        (config.seed + round_number) % (2**31),
        covariances=policy["covariances"],
    )
    return select_diverse(
        embeddings,
        scores,
        policy["eligible"],
        policy["means"],
        policy["sigmas"],
        config.diversity,
        config.batch_size,
    )


def evaluate_agent(
    policy: dict,
    observed: np.ndarray,
    common_unseen: np.ndarray,
    oracle: np.ndarray,
    normalized_oracle: np.ndarray,
    top_mask: np.ndarray,
    last_batch: list[int],
    ids: list[str],
) -> dict:
    """Metrics are retrospective only and never flow back to either selection policy."""
    hits = int(top_mask[observed].sum())
    fraction = float(top_mask.mean())
    return {
        "observed_count": len(observed),
        "top_hits": hits,
        "top_k_recall": 100 * hits / int(top_mask.sum()),
        "enrichment": (hits / len(observed)) / fraction,
        "spearman_rho": spearman(policy["means"][common_unseen, 0], oracle[common_unseen, 0]),
        "hypervolume": hypervolume(normalized_oracle[observed], np.array([-0.05, -0.05])),
        "feasible_candidates": int(policy["eligible"].sum()),
        "selected_ids": [ids[i] for i in last_batch],
        "observed_ids": [ids[i] for i in observed],
    }


def run_benchmark(config: BenchmarkConfig, progress: Callable[[dict], None] = lambda _: None) -> dict:
    """Advance two independent agents concurrently; initial seeds and budget match.

    Round zero is the shared seed. Every later point is measured AFTER the new
    batch is revealed and models refit. Spearman uses the same still-unobserved
    intersection for both agents at that round (its membership changes by round).
    """
    started = time.perf_counter()
    data = load_dataset(config.dataset_id)
    n = len(data.ids)
    validate_budget(config, n)
    embeddings = fragment_mutation_embeddings(data.sequences, data.metadata["parent_sequence"])
    streams = np.random.SeedSequence(config.seed).spawn(3)
    seeds = np.sort(np.random.default_rng(streams[0]).choice(n, config.seed_size, replace=False))
    observed = {agent: seeds.copy() for agent in ("active", "random")}
    generators = {agent: np.random.default_rng(streams[i + 1]) for i, agent in enumerate(observed)}
    top_count = max(1, math.ceil(0.05 * n))
    cutoff = float(np.sort(data.oracle[:, 0])[-top_count])
    top_mask = data.oracle[:, 0] >= cutoff
    # Fixed oracle normalization is confined to retrospective metric computation.
    lower = data.oracle.min(0)
    scale = np.maximum(np.ptp(data.oracle, axis=0), 1e-12)
    normalized = (data.oracle - lower) / scale
    result = {
        "config": config.model_dump(),
        "dataset": data.metadata,
        "rounds": [],
        "seed_ids": [data.ids[i] for i in seeds],
        "status": "running",
        "stop_reason": None,
        "top_k": {
            "nominal_count": top_count,
            "actual_count_including_ties": int(top_mask.sum()),
            "cutoff_binding": cutoff,
            "population_size": n,
        },
        "protocol": {
            "embedding": "Exact position/residue mutation contrasts plus CDR-specific mutation counts; no invented framework or pLM context",
            "surrogate": "Joint two-output PyTorch exact GP with learned binding-expression covariance",
            "feasibility": "Each agent's predicted expression-proxy mean minus beta*sigma meets the same floor; no oracle prefilter",
            "correlation_population": "Common still-unmeasured intersection, same IDs for both agents within a round; changing cohort across rounds",
            "hypervolume": {
                "normalization_min": lower.tolist(),
                "normalization_scale": scale.tolist(),
                "reference": [-0.05, -0.05],
                "oracle_maximum": hypervolume(normalized, np.array([-0.05, -0.05])),
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    last_batches = {agent: seeds.tolist() for agent in observed}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="benchmark-agent") as workers:
        for round_number in range(config.rounds + 1):
            result["progress"] = f"Fitting both agents after round {round_number}/{config.rounds}"
            progress(result)
            futures = {
                agent: workers.submit(
                    fit_policy, embeddings, indices.copy(), data.oracle[indices].copy(), config
                )
                for agent, indices in observed.items()
            }
            policies = {agent: future.result() for agent, future in futures.items()}
            common = policies["active"]["unseen"] & policies["random"]["unseen"]
            record = {
                "round": round_number,
                "evaluation_count": int(common.sum()),
                "evaluation_ids": [data.ids[i] for i in np.flatnonzero(common)],
            }
            for agent in observed:
                record[agent] = evaluate_agent(
                    policies[agent],
                    observed[agent],
                    common,
                    data.oracle,
                    normalized,
                    top_mask,
                    last_batches[agent],
                    data.ids,
                )
                logger.info(
                    "Benchmark seed=%s round=%s agent=%s recall=%.3f rho=%s hypervolume=%.6f",
                    config.seed,
                    round_number,
                    agent,
                    record[agent]["top_k_recall"],
                    record[agent]["spearman_rho"],
                    record[agent]["hypervolume"],
                )
            result["rounds"].append(record)
            progress(result)
            if round_number == config.rounds:
                result["status"] = "completed"
                break
            try:
                # Plan both batches before revealing either to preserve equal budgets.
                batches = {
                    agent: choose_batch(
                        policies[agent],
                        embeddings,
                        data.oracle[indices].copy(),
                        config,
                        generators[agent],
                        agent,
                        round_number + 1,
                    )
                    for agent, indices in observed.items()
                }
            except ValueError as exc:
                result["status"] = "stopped_early"
                result["stop_reason"] = str(exc)
                break
            for agent, batch in batches.items():
                observed[agent] = np.r_[observed[agent], batch]
            last_batches = batches
    final = result["rounds"][-1]
    active, random = final["active"], final["random"]
    result["summary"] = {
        "completed_rounds": final["round"],
        "assays_per_agent": active["observed_count"],
        "active_enrichment": active["enrichment"],
        "random_enrichment": random["enrichment"],
        "recall_multiplier": active["top_k_recall"] / random["top_k_recall"]
        if random["top_k_recall"] > 0
        else None,
        "recall_difference_pp": active["top_k_recall"] - random["top_k_recall"],
        "correlation_difference": active["spearman_rho"] - random["spearman_rho"]
        if active["spearman_rho"] is not None and random["spearman_rho"] is not None
        else None,
        "hypervolume_difference": active["hypervolume"] - random["hypervolume"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    result["progress"] = (
        "Simulation complete"
        if result["status"] == "completed"
        else "Stopped: insufficient feasible candidates"
    )
    return result
