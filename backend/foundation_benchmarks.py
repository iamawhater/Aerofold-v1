"""Frozen ESM-2-only and AntiBERTy-only supervised benchmark comparators.

Both use mean-pooled residue embeddings (special tokens excluded), heavy/light
concatenation and the same validation-tuned ridge head. Neither includes
mutation descriptors, the AeroFold ensemble, or masked-likelihood penalties.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from backend.predictions import PredictionBenchmarkConfig, benchmark_plans, metrics, run_prediction_benchmark
from backend.public_data import fingerprint, load_task

CHECKPOINTS = {
    "esm2-only": {
        "model": "facebook/esm2_t6_8M_UR50D",
        "revision": "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
        "dimension": 320,
        "package": "transformers",
    },
    "antiberty-only": {
        "model": "jeffruffolo/AntiBERTy",
        "revision": "d582006e0c7dc6917d26fa60740a357e8df7b466",
        "dimension": 512,
        "package": "antiberty",
    },
}
PREDICTORS = ("ensemble", *CHECKPOINTS)
POOLING = "last-layer-residue-mean-heavy-light-concatenation-v1"


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pool_residues(states: torch.Tensor, sequence: str) -> np.ndarray:
    """Mean over residues only; AntiBERTy tensors include one token at each end."""
    if states.ndim != 2 or states.shape[0] != len(sequence) + 2:
        raise ValueError("Model returned unexpected residue/token alignment")
    vector = states[1:-1].mean(0).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Model returned nonfinite residue embeddings")
    return vector


def frozen_embeddings(
    records: list[dict],
    predictor: str,
    root: Path,
    progress: Callable[[str], None],
    batch_size: int = 16,
    local_only: bool = False,
) -> tuple[np.ndarray, dict]:
    """Encode each unique chain once, using pinned safe checkpoints and scoped caches."""
    spec = CHECKPOINTS[predictor]
    try:
        from huggingface_hub import snapshot_download

        if predictor == "antiberty-only":
            from antiberty import AntiBERTyRunner
        else:
            from transformers import AutoTokenizer, EsmModel
    except ImportError as exc:
        raise ValueError(
            "Install benchmark encoders: pip install -r requirements-benchmark-models.txt"
        ) from exc
    progress(f"Loading {predictor}: {spec['model']} at {spec['revision']}")
    patterns = ["config.json", "model.safetensors"]
    if predictor == "esm2-only":
        patterns += ["vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]
    snapshot = Path(
        snapshot_download(
            spec["model"],
            revision=spec["revision"],
            cache_dir=root / "huggingface/hub",
            allow_patterns=patterns,
            local_files_only=local_only,
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if predictor == "esm2-only":
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        model = (
            EsmModel.from_pretrained(
                snapshot, local_files_only=True, use_safetensors=True, add_pooling_layer=False
            )
            .to(device)
            .eval()
        )
    else:
        runner = AntiBERTyRunner(device=str(device), checkpoint_path=str(snapshot))
    package_version = importlib.metadata.version(spec["package"])
    # Include actual checkpoint bytes and pooling in cache identity.
    checkpoint_hash = _sha(snapshot / "model.safetensors")
    namespace = f"{spec['model']}:{spec['revision']}:{checkpoint_hash}:{package_version}:{POOLING}"
    cache = root / "benchmark-embedding-cache" / predictor
    cache.mkdir(parents=True, exist_ok=True)
    chains = sorted(
        {r.get(chain, "") for r in records for chain in ("heavy_sequence", "light_sequence")} - {""}
    )
    vectors, pending = {}, []
    for sequence in chains:
        key = hashlib.sha256((namespace + ":" + sequence).encode()).hexdigest()
        path = cache / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                vector = saved["embedding"]
            if vector.shape != (spec["dimension"],) or not np.isfinite(vector).all():
                raise ValueError("Invalid cached embedding; remove the affected cache entry and retry")
            vectors[sequence] = vector
        else:
            pending.append((sequence, path))
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        sequences = [sequence for sequence, _ in batch]
        progress(
            f"{predictor}: encoding chains {start + 1}–{min(start + batch_size, len(pending))} / {len(pending)} ({len(chains) - len(pending)} cached)"
        )
        with torch.inference_mode():
            if predictor == "esm2-only":
                tokens = tokenizer(
                    sequences, padding=True, return_tensors="pt", return_special_tokens_mask=True
                )
                special = tokens.pop("special_tokens_mask").to(device).bool()
                tokens = {name: value.to(device) for name, value in tokens.items()}
                states = model(**tokens).last_hidden_state
                residue_mask = tokens["attention_mask"].bool() & ~special
                pooled = (states * residue_mask.unsqueeze(-1)).sum(1) / residue_mask.sum(1, keepdim=True)
                output = pooled.cpu().numpy().astype(np.float64)
            else:
                output = np.stack(
                    [
                        pool_residues(state, sequence)
                        for state, sequence in zip(runner.embed(sequences, hidden_layer=-1), sequences)
                    ]
                )
        if output.shape != (len(batch), spec["dimension"]) or not np.isfinite(output).all():
            raise ValueError("Invalid encoder output")
        for (sequence, path), vector in zip(batch, output):
            vectors[sequence] = vector
            np.savez_compressed(path, embedding=vector)
    zero = np.zeros(spec["dimension"])
    features = np.stack(
        [
            np.concatenate(
                [
                    vectors.get(r.get("heavy_sequence", ""), zero),
                    vectors.get(r.get("light_sequence", ""), zero),
                ]
            )
            for r in records
        ]
    )
    provenance = dict(
        predictor=predictor,
        **spec,
        package_version=package_version,
        weights_sha256=checkpoint_hash,
        config_sha256=_sha(snapshot / "config.json"),
        pooling=POOLING,
        missing_chain="Zero-filled chain block; no synthetic sequence",
        dimensions=2 * spec["dimension"],
        device=str(device),
        frozen=True,
        features="Foundation embeddings only; no mutation descriptors or ensemble blend",
    )
    return features, provenance


class EmbeddingOnlyRegressor:
    """A single ridge head, selected on validation and calibrated independently."""

    def __init__(self, seed: int, predictor: str):
        self.seed, self.predictor = seed, predictor

    @staticmethod
    def _model(alpha: float):
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-5))

    def fit(self, embeddings: np.ndarray, scores: np.ndarray, ids: list[str]) -> "EmbeddingOnlyRegressor":
        """Match the ensemble's fitting/validation/calibration partition exactly."""
        if len(scores) < 60 or len(embeddings) != len(scores) or len(ids) != len(scores):
            raise ValueError("At least 60 aligned unique labeled examples are required")
        if not np.isfinite(embeddings).all() or not np.isfinite(scores).all():
            raise ValueError("All embeddings and labels must be finite")
        order = np.random.default_rng(self.seed).permutation(len(scores))
        ncal = max(12, math.ceil(0.2 * len(scores)))
        nval = max(12, math.ceil(0.2 * len(scores)))
        cal, val, train = order[:ncal], order[ncal : ncal + nval], order[ncal + nval :]
        mean, scale = float(scores[train].mean()), max(float(scores[train].std()), 1e-8)
        standardized = (scores - mean) / scale
        with threadpool_limits(limits=2):
            candidates = []
            for alpha in (1, 10, 100, 1000):
                model = self._model(alpha).fit(embeddings[train], standardized[train])
                pred = model.predict(embeddings[val])
                candidates.append((float(np.mean((pred - standardized[val]) ** 2)), alpha))
            loss, self.alpha = min(candidates)
            development = np.concatenate([train, val])
            self.mean = float(scores[development].mean())
            self.scale = max(float(scores[development].std()), 1e-8)
            self.model = self._model(self.alpha).fit(
                embeddings[development], (scores[development] - self.mean) / self.scale
            )
            residual = np.sort(np.abs(scores[cal] - self._predict(embeddings[cal])))
        self.radius = float(residual[min(len(residual), math.ceil((len(residual) + 1) * 0.9)) - 1])
        self.validation_rmse = math.sqrt(loss) * scale
        self.fitting_ids = [ids[i] for i in development]
        self.validation_ids = [ids[i] for i in val]
        self.calibration_ids = [ids[i] for i in cal]
        return self

    def _predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.model.predict(embeddings) * self.scale + self.mean

    def predict(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        with threadpool_limits(limits=2):
            pred = self._predict(embeddings)
        return dict(prediction=pred, lower=pred - self.radius, upper=pred + self.radius)

    def card(self) -> dict:
        return dict(
            predictor=self.predictor,
            head="StandardScaler + supervised ridge; frozen embeddings only",
            alpha=self.alpha,
            alpha_grid=[1, 10, 100, 1000],
            validation_rmse=self.validation_rmse,
            fitting_ids=self.fitting_ids,
            validation_ids=self.validation_ids,
            calibration_ids=self.calibration_ids,
            conformal_radius=self.radius,
            nominal_coverage=0.9,
        )


def run_benchmark_suite(
    config: PredictionBenchmarkConfig,
    root: Path,
    predictors: tuple[str, ...] = ("ensemble",),
    progress: Callable[[str], None] = lambda _: None,
    batch_size: int = 16,
    local_only: bool = False,
) -> dict:
    """Run requested models on identical outer/inner splits and merge their audits."""
    if not predictors or len(set(predictors)) != len(predictors) or set(predictors) - set(PREDICTORS):
        raise ValueError("Choose unique predictors from ensemble, esm2-only, antiberty-only")
    if not 1 <= batch_size <= 128:
        raise ValueError("Embedding batch size must be between 1 and 128")
    started = time.monotonic()
    metadata, records = load_task(config.task_id, root / "custom-tasks")
    plans = benchmark_plans(config, records)
    union = sorted({int(i) for _, development, test, _ in plans for i in [*development, *test]})
    lookup = {index: pos for pos, index in enumerate(union)}
    if "ensemble" in predictors:
        result = run_prediction_benchmark(config, root, progress)
    else:
        result = dict(
            kind="prediction_benchmark",
            task=metadata,
            config=config.model_dump(),
            curves=[],
            runs=[],
            split_audit=[],
            protocol="Duplicate sequence pairs aggregated before splitting. Hidden test identities are fixed across budgets within each seed. Hyperparameters use internal validation only; conformal intervals use separate calibration labels.",
            limitations=[
                "Random library splits measure within-assay interpolation, not transfer to new antigens.",
                "Foundation pretraining overlap and censored assay endpoints remain unresolved.",
            ],
        )
        for seed, development, test, _ in plans:
            result["split_audit"].append(
                dict(
                    seed=seed,
                    training_ids=[records[i]["variant_id"] for i in development],
                    test_ids=[records[i]["variant_id"] for i in test],
                    test_hash=hashlib.sha256(
                        "|".join(sorted(fingerprint(records[i]) for i in test)).encode()
                    ).hexdigest(),
                )
            )
    runs = {(r["seed"], r["training_size"]): r for r in result["runs"]}
    curves = {c["training_size"]: c for c in result["curves"]}
    provenance = {}
    for predictor in predictors:
        if predictor == "ensemble":
            continue
        encoded, provenance[predictor] = frozen_embeddings(
            [records[i] for i in union], predictor, root, progress, batch_size, local_only
        )

        def embedding(ids):
            return encoded[[lookup[int(i)] for i in ids]]

        for seed, development, test, sizes in plans:
            truth = np.array([records[i]["score"] for i in test])
            for size in sizes:
                chosen = development[:size]
                progress(f"{predictor}: seed {seed}, {size} labeled sequences, {len(test)} hidden tests")
                model = EmbeddingOnlyRegressor(seed, predictor).fit(
                    embedding(chosen),
                    np.array([records[i]["score"] for i in chosen]),
                    [records[i]["variant_id"] for i in chosen],
                )
                pred = model.predict(embedding(test))
                item = runs.setdefault((seed, size), dict(seed=seed, training_size=size))
                item[predictor] = metrics(
                    truth,
                    pred["prediction"],
                    metadata["maximize"],
                    config.top_k,
                    pred["lower"],
                    pred["upper"],
                )
                item.setdefault("predictor_cards", {})[predictor] = model.card()
        for size in sorted({size for _, _, _, sizes in plans for size in sizes}):
            entries = [r[predictor] for (_, budget), r in runs.items() if budget == size]
            point = curves.setdefault(size, dict(training_size=size, repeats=len(entries)))
            point[predictor] = {
                key: float(np.mean([r[key] for r in entries if r[key] is not None]))
                if any(r[key] is not None for r in entries)
                else None
                for key in entries[0]
            }
            point.setdefault("predictor_recall_ranges", {})[predictor] = [
                min(r["top_recall"] for r in entries),
                max(r["top_recall"] for r in entries),
            ]
    result.update(
        curves=[curves[size] for size in sorted(curves)],
        runs=[runs[key] for key in sorted(runs)],
        encoder_provenance=provenance,
        predictors=list(predictors),
        seconds=time.monotonic() - started,
    )
    return result
