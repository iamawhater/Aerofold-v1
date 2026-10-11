"""Auditable 2D Pareto optimization, marginal MC-EHVI and quality-weighted DPP."""

from __future__ import annotations

import numpy as np


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Keep nondominated points under maximization, including exact ties."""
    return np.array(
        [not np.any(np.all(points >= p, axis=1) & np.any(points > p, axis=1)) for p in points], dtype=bool
    )


def hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    valid = points[np.all(points > reference, axis=1)]
    if not len(valid):
        return 0.0
    frontier = valid[pareto_mask(valid)]
    frontier = frontier[np.argsort(frontier[:, 0])]
    widths = np.diff(np.r_[reference[0], frontier[:, 0]])
    return float(np.sum(widths * (frontier[:, 1] - reference[1])))


def expected_hv_improvement(
    mean: np.ndarray,
    sigma: np.ndarray,
    baseline: np.ndarray,
    seed: int,
    samples: int = 256,
    covariance: np.ndarray | None = None,
) -> np.ndarray:
    """Marginal MC-EHVI with optional correlated Gaussian objective draws.

    Normalize using measured outcomes; integrate the uncovered area of each
    posterior sample against disjoint rectangles of the observed frontier.
    """
    lower = baseline.min(0)
    scale = np.maximum(np.ptp(baseline, axis=0), 1.0)
    normalized = (baseline - lower) / scale
    reference = np.array([-0.15, -0.15])
    front = normalized[pareto_mask(normalized)]
    front = front[np.argsort(front[:, 0])]
    rng = np.random.default_rng(seed)
    # Common random numbers reduce Monte Carlo ranking variance.
    z = rng.standard_normal((samples, 1, 2))
    if covariance is None:
        perturbations = z * sigma[None, :, :]
    else:
        if covariance.shape != (len(mean), 2, 2):
            raise ValueError("Expected one 2x2 objective covariance per candidate")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 1e-12))[:, None, :]) @ np.swapaxes(
            eigenvectors, -1, -2
        )
        perturbations = np.einsum("sni,nij->snj", np.broadcast_to(z, (samples, len(mean), 2)), root)
    draws = ((mean[None, :, :] + perturbations) - lower) / scale
    area = np.maximum(draws[..., 0] - reference[0], 0) * np.maximum(draws[..., 1] - reference[1], 0)
    previous = reference[0]
    for x, y in front:
        area -= np.maximum(np.minimum(draws[..., 0], x) - previous, 0) * np.maximum(
            np.minimum(draws[..., 1], y) - reference[1], 0
        )
        previous = x
    return np.maximum(area, 0).mean(0)


def rank_scale(values: np.ndarray) -> np.ndarray:
    """Tie-aware empirical percentile scaling."""
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) == 1:
        return np.full(len(values), 0.5)
    ranks = np.cumsum(counts) - (counts + 1) / 2
    return ranks[inverse] / max(len(values) - 1, 1)


def select_diverse(
    embeddings: np.ndarray,
    acquisition: np.ndarray,
    eligible: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    diversity: float,
    batch_size: int = 24,
) -> list[int]:
    """Greedy DPP MAP on a frontier-enriched acquisition shortlist.

    The strict feasibility mask is applied first and never relaxed. The pool
    includes mean/UCB frontiers and at least 4 batches by acquisition score.
    """
    indices = np.flatnonzero(eligible)
    if len(indices) < batch_size:
        raise ValueError(
            f"Only {len(indices)} unmeasured variants satisfy all constraints; exactly {batch_size} are required. Review stability confidence and liability thresholds."
        )
    ranked = indices[np.argsort(-acquisition[indices], kind="stable")]
    mean_front = indices[pareto_mask(means[indices])]
    ucb_front = indices[pareto_mask(means[indices] + sigmas[indices])]
    pool = np.unique(np.concatenate([ranked[: max(96, batch_size * 4)], mean_front, ucb_front]))
    if diversity == 0:
        return ranked[:batch_size].tolist()
    x = embeddings[pool]
    scale = np.maximum(embeddings.std(0), 0.05)
    x = x / scale
    distances = (
        np.maximum(np.sum(x * x, axis=1)[:, None] + np.sum(x * x, axis=1)[None, :] - 2 * x @ x.T, 0)
        / x.shape[1]
    )
    positive = distances[distances > 1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    similarity = np.exp(-distances / (2 * max(bandwidth, 1e-6)))
    kernel = diversity * similarity + (1 - diversity + 1e-8) * np.eye(len(pool))
    quality = np.exp(2 * rank_scale(acquisition[pool]))
    kernel *= quality[:, None] * quality[None, :]
    residual = np.diag(kernel).copy()
    projections = np.zeros((batch_size, len(pool)))
    chosen = []
    for step in range(batch_size):
        best = int(np.argmax(residual))
        chosen.append(best)
        projections[step] = (kernel[best] - projections[:step, best] @ projections[:step]) / np.sqrt(
            max(residual[best], 1e-14)
        )
        residual = np.maximum(residual - projections[step] ** 2, 0)
        residual[chosen] = -np.inf
    return pool[chosen].tolist()
