"""Validation-selected sequence ensemble with independent split-conformal calibration.

AbLang-2 is a frozen antibody representation, not a pretrained affinity oracle.
All supervised heads, feature reference states and hyperparameters are learned
from this assay's development data. Test labels never enter this module.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from backend.features import AbLang2Embedder
from backend.models import SequenceRecord

AA = "-ACDEFGHIKLMNPQRSTVWY"
CODE = {aa: i for i, aa in enumerate(AA)}
VERSION = "supervised-ensemble-1"


def residue_codes(records: list[dict]) -> np.ndarray:
    """Chain-separated, padded residue identities; no invented antibody numbering."""
    result = np.zeros((len(records), 360), dtype=np.uint8)
    for i, row in enumerate(records):
        for offset, chain in ((0, "heavy_sequence"), (180, "light_sequence")):
            sequence = row.get(chain, "")
            result[i, offset : offset + len(sequence)] = [CODE[aa] for aa in sequence]
    return result


class MutationFeatures:
    """Sparse residue contrasts against a training-only modal sequence."""

    def fit(self, codes: np.ndarray) -> "MutationFeatures":
        self.reference = np.array([np.bincount(col, minlength=21).argmax() for col in codes.T])
        return self

    def transform(self, codes: np.ndarray) -> sparse.csr_matrix:
        rows, positions = np.where(codes != self.reference)
        columns = positions * 21 + codes[rows, positions]
        ref_columns = positions * 21 + self.reference[positions]
        return sparse.csr_matrix(
            (
                np.concatenate([np.ones(len(rows)), -np.ones(len(rows))]),
                (np.concatenate([rows, rows]), np.concatenate([columns, ref_columns])),
            ),
            shape=(len(codes), 360 * 21),
        )


def antibody_embeddings(records: list[dict], cache: Path, progress: Callable[[str], None]) -> np.ndarray:
    """Encode sequences only, with honest progress and the actual pretrained model."""
    progress("Loading pretrained paired AbLang-2")
    encoder = AbLang2Embedder(cache)
    batches = []
    for start in range(0, len(records), 128):
        progress(f"AbLang-2: encoding {start + 1}–{min(start + 128, len(records))} of {len(records)} domains")
        batches.append(
            encoder.encode(
                [
                    SequenceRecord(
                        **{key: r.get(key, "") for key in ("variant_id", "heavy_sequence", "light_sequence")}
                    )
                    for r in records[start : start + 128]
                ]
            )
        )
    return np.vstack(batches)


def nearest_identity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Conservative unaligned positional identity; length changes reduce support."""
    result = np.zeros(len(query))
    for i in range(0, len(query), 16):
        block = query[i : i + 16]
        best = np.zeros(len(block))
        for j in range(0, len(reference), 256):
            train = reference[j : j + 256]
            present = (block[:, None] != 0) | (train[None] != 0)
            same = (block[:, None] == train[None]) & present
            similarity = same.sum(axis=2) / np.maximum(1, present.sum(axis=2))
            best = np.maximum(best, similarity.max(axis=1))
        result[i : i + len(block)] = best
    return result


class PredictionEnsemble:
    """Additive, epistatic, tree and optional antibody-foundation-model heads."""

    def __init__(self, seed: int = 42, coverage: float = 0.9):
        self.seed, self.coverage = seed, coverage

    @staticmethod
    def _head(name: str, alpha: float, count: int, gamma: float, seed: int):
        if name == "additive":
            return Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
        if name == "ablang2":
            return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-5))
        if name == "forest":
            return ExtraTreesRegressor(
                n_estimators=160, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=2
            )
        kernel = "rbf" if name == "local_kernel" else "polynomial"
        degree = 2 if name == "pair_interactions" else 3
        return make_pipeline(
            Nystroem(
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=1,
                n_components=min(256, count),
                random_state=seed,
            ),
            Ridge(alpha=alpha),
        )

    def fit(
        self,
        records: list[dict],
        scores: np.ndarray,
        embeddings: np.ndarray | None = None,
        progress: Callable[[str], None] = lambda _: None,
    ) -> "PredictionEnsemble":
        """Select on validation, refit on development, calibrate on untouched labels."""
        if len(records) < 60:
            raise ValueError("At least 60 unique training sequence pairs are required")
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite training labels")
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(records))
        ncal = max(12, math.ceil(0.2 * len(records)))
        nval = max(12, math.ceil(0.2 * len(records)))
        cal, val, train = indices[:ncal], indices[ncal : ncal + nval], indices[ncal + nval :]
        codes = residue_codes(records)
        features = MutationFeatures().fit(codes[train])
        x = features.transform(codes)
        gamma = 1 / max(2, float(x[train].multiply(x[train]).sum(axis=1).mean()))
        ymean, yscale = float(scores[train].mean()), max(float(scores[train].std()), 1e-8)
        y = (scores - ymean) / yscale
        names = ["additive", "pair_interactions", "higher_interactions", "local_kernel", "forest"]
        if embeddings is not None:
            names.append("ablang2")
        best_alpha, predictions, errors = {}, [], []
        with threadpool_limits(limits=2):
            for name in names:
                progress(f"Validating {name.replace('_', ' ')} on this assay")
                source = embeddings if name == "ablang2" else x
                alphas = (
                    [1.0]
                    if name == "forest"
                    else ([1, 10, 100, 1000] if name == "ablang2" else [0.01, 0.1, 1, 10])
                )
                candidates = []
                # Nyström basis is independent of alpha; reuse it across the ridge path.
                head = self._head(name, alphas[0], len(train), gamma, self.seed)
                if name in {"pair_interactions", "higher_interactions", "local_kernel"}:
                    basis = head[0]
                    xt = basis.fit_transform(source[train])
                    xv = basis.transform(source[val])
                    for alpha in alphas:
                        candidate = Ridge(alpha=alpha).fit(xt, y[train])
                        pred = candidate.predict(xv)
                        candidates.append((float(np.mean((pred - y[val]) ** 2)), alpha, pred))
                else:
                    for alpha in alphas:
                        candidate = self._head(name, alpha, len(train), gamma, self.seed).fit(
                            source[train], y[train]
                        )
                        pred = candidate.predict(source[val])
                        candidates.append((float(np.mean((pred - y[val]) ** 2)), alpha, pred))
                loss, alpha, pred = min(candidates, key=lambda item: item[0])
                best_alpha[name] = alpha
                predictions.append(pred)
                errors.append(loss)
            matrix = np.column_stack(predictions)
            best = np.zeros(len(names))
            best[int(np.argmin(errors))] = 1

            # Regularization toward the best individual model tempers small-validation overfitting.
            def objective(w: np.ndarray) -> float:
                return float(np.mean((matrix @ w - y[val]) ** 2) + 0.02 * np.sum((w - best) ** 2))

            fit = minimize(
                objective,
                best,
                bounds=[(0, 1)] * len(names),
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                method="SLSQP",
            )
            weights = fit.x if fit.success and np.isfinite(fit.x).all() else best
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            self.names, self.weights = names, weights
            self.validation = {
                name: {"rmse": float(np.sqrt(error) * yscale), "weight": float(w), "alpha": best_alpha[name]}
                for name, error, w in zip(names, errors, weights)
            }
            development = np.concatenate([train, val])
            self.features = MutationFeatures().fit(codes[development])
            x = self.features.transform(codes)
            self.mean, self.scale = (
                float(scores[development].mean()),
                max(float(scores[development].std()), 1e-8),
            )
            y = (scores - self.mean) / self.scale
            gamma = 1 / max(2, float(x[development].multiply(x[development]).sum(axis=1).mean()))
            self.models = {}
            for name in names:
                progress(f"Fitting final {name.replace('_', ' ')} head")
                source = embeddings if name == "ablang2" else x
                self.models[name] = self._head(
                    name, best_alpha[name], len(development), gamma, self.seed
                ).fit(source[development], y[development])
            self.reference = codes[development]
            prediction = self._predict(codes[cal], embeddings[cal] if embeddings is not None else None)[
                "prediction"
            ]
        residual = np.sort(np.abs(scores[cal] - prediction))
        rank = min(len(residual), math.ceil((len(residual) + 1) * self.coverage))
        self.radius = float(residual[rank - 1])
        self.counts = {
            "fitting": len(development),
            "validation_before_refit": len(val),
            "calibration": len(cal),
            "total_labeled": len(records),
        }
        self.calibration_ids = [records[i]["variant_id"] for i in cal]
        return self

    def _predict(self, codes: np.ndarray, embeddings: np.ndarray | None) -> dict[str, np.ndarray]:
        x = self.features.transform(codes)
        predictions = (
            np.column_stack(
                [self.models[name].predict(embeddings if name == "ablang2" else x) for name in self.names]
            )
            * self.scale
            + self.mean
        )
        mean = predictions @ self.weights
        # Dispersion across models is a diagnostic, not a calibrated epistemic posterior.
        disagreement = np.sqrt(np.mean((predictions - mean[:, None]) ** 2, axis=1))
        return {
            "prediction": mean,
            "disagreement": disagreement,
            "additive": predictions[:, self.names.index("additive")],
            "mean_baseline": np.full(len(codes), self.mean),
        }

    def predict(
        self, records: list[dict], embeddings: np.ndarray | None = None, support: bool = True
    ) -> dict[str, np.ndarray]:
        """Return native assay predictions, calibrated intervals and support diagnostics."""
        codes = residue_codes(records)
        with threadpool_limits(limits=2):
            result = self._predict(codes, embeddings)
        result["lower"] = result["prediction"] - self.radius
        result["upper"] = result["prediction"] + self.radius
        if support:
            result["nearest_identity"] = nearest_identity(codes, self.reference)
        return result

    def card(self) -> dict:
        """Serializable, inspectable training and uncertainty provenance."""
        return dict(
            version=VERSION,
            model="Validation-selected supervised sequence ensemble",
            foundation_model="AbLang-2 paired / package 0.1.1" if "ablang2" in self.names else None,
            heads=self.validation,
            training=self.counts,
            nominal_coverage=self.coverage,
            conformal_radius=self.radius,
            uncertainty="Split-conformal absolute-residual interval; model dispersion is not calibrated epistemic uncertainty",
            calibration_ids=self.calibration_ids,
            limitations=[
                "Coverage assumes exchangeability; not guaranteed on novel scaffolds or new antigens.",
                "Position features are not IMGT-aligned; indels and distant scaffolds require special validation.",
                "Frozen foundation-model pretraining overlap is unknown.",
                "Validation refitting changes heads; interval calibration remains independent.",
            ],
        )
