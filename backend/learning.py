"""Shared GP fitting and acquisition scoring for discovery and benchmarking."""

import numpy as np

from backend.optimization import expected_hv_improvement, rank_scale
from backend.surrogate import CorrelatedGP


def fit_objectives(
    embeddings: np.ndarray, training: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit paired objectives using only revealed observations."""
    model = CorrelatedGP().fit(embeddings[training], targets)
    return model.predict(embeddings)


def acquisition_scores(
    means: np.ndarray,
    sigmas: np.ndarray,
    baseline: np.ndarray,
    unmeasured: np.ndarray,
    exploration: float,
    seed: int,
    uncertainty_scale: np.ndarray | None = None,
    covariances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """MC-EHVI percentile blended with a training-size-decayed uncertainty bonus."""
    ehvi = expected_hv_improvement(means, sigmas, baseline, seed, covariance=covariances)
    scale = baseline.std(0) if uncertainty_scale is None else uncertainty_scale
    uncertainty = np.sqrt(np.sum((sigmas / np.maximum(scale, 0.1)) ** 2, axis=1))
    weight = exploration / (1 + len(baseline) / 100)
    scores = np.zeros(len(means))
    scores[unmeasured] = (1 - weight) * rank_scale(ehvi[unmeasured]) + weight * rank_scale(
        uncertainty[unmeasured]
    )
    return scores, ehvi, weight
