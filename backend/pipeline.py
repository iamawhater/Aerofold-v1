"""End-to-end selection using measured outcomes only; no oracle imports."""

from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from backend.features import AbLang2Embedder, ESMEmbedder, descriptor_embeddings, developability
from backend.learning import acquisition_scores, fit_objectives
from backend.models import DatasetInput, SelectionConfig
from backend.optimization import pareto_mask, select_diverse
from backend.sequences import annotate, mutations


def run_selection(
    dataset: DatasetInput,
    config: SelectionConfig,
    cache_dir: Path,
    progress: Callable[[str], None] = lambda _: None,
) -> dict:
    """Train surrogates, filter, optimize and serialize an immutable recommendation."""
    started = time.perf_counter()
    if len(dataset.observations) < 6:
        raise ValueError(
            "At least 6 paired binding/stability observations are required. Upload measurements before fitting; sequences alone cannot determine affinity."
        )
    records = [dataset.parent, *dataset.variants]
    observations = {o.variant_id: o for o in dataset.observations}
    annotations = {
        chain: annotate(seq, chain)
        for chain, seq in (("H", dataset.parent.heavy_sequence), ("L", dataset.parent.light_sequence))
        if seq
    }
    progress(
        "Encoding sequences"
        if config.embedding == "descriptor"
        else "Encoding ESM-2 and masked likelihoods; this can take several minutes"
        if config.embedding == "esm2"
        else "Encoding full antibody domains with AbLang-2; first run can take several minutes"
    )
    if config.embedding == "esm2":
        embeddings, perplexities = ESMEmbedder(cache_dir).encode(records)
    elif config.embedding == "ablang2":
        embeddings = AbLang2Embedder(cache_dir).encode(records)
        perplexities = [None] * len(records)
    else:
        embeddings = descriptor_embeddings(records)
        perplexities = [None] * len(records)
    training = np.array([i for i, v in enumerate(records) if v.variant_id in observations])
    # Bound exact-GP cubic cost for service responsiveness. Record subsampling.
    if len(training) > 512:
        training = training[np.linspace(0, len(training) - 1, 512, dtype=int)]
    targets = np.array(
        [
            [observations[records[i].variant_id].binding, observations[records[i].variant_id].stability]
            for i in training
        ]
    )
    progress(f"Fitting joint two-objective Gaussian process on {len(training)} observations")
    means, sigmas, covariances = fit_objectives(embeddings, training, targets)
    raw_stability = means[:, 1].copy()
    properties = [developability(r) for r in records]
    penalties = np.array(
        [
            3 * max(0, p["hydrophobic_patch"] - 0.65) + 2 * p["apr"] + 0.1 * max(0, p["charge_patch"] - 3)
            for p in properties
        ]
    )
    if config.embedding == "esm2":
        penalties += 2 * np.maximum(0, np.log(np.array(perplexities, dtype=float) / float(perplexities[0])))
    means[:, 1] -= penalties
    adjusted_baseline = targets.copy()
    adjusted_baseline[:, 1] -= penalties[training]
    unmeasured = np.array([r.variant_id not in observations and i > 0 for i, r in enumerate(records)])
    lcb = means[:, 1] - config.confidence_beta * sigmas[:, 1]
    stability_ok = lcb >= config.min_stability
    charge_ok = np.array([p["max_chain_abs_charge"] <= config.max_abs_charge for p in properties])
    hydro_ok = np.array([p["hydrophobic_patch"] <= config.max_hydrophobic_patch for p in properties])
    apr_ok = np.array([p["apr"] <= config.max_apr for p in properties])
    eligible = unmeasured & stability_ok & charge_ok & hydro_ok & apr_ok
    progress("Calculating expected hypervolume improvement and selecting 24 diverse variants")
    acquisition, ehvi, exploration_weight = acquisition_scores(
        means,
        sigmas,
        adjusted_baseline,
        unmeasured,
        config.exploration,
        config.seed,
        uncertainty_scale=targets.std(0),
        covariances=covariances,
    )
    selected = select_diverse(embeddings, acquisition, eligible, means, sigmas, config.diversity)
    candidate_indices = np.flatnonzero(unmeasured)
    pareto = np.zeros(len(records), dtype=bool)
    pareto[candidate_indices] = pareto_mask(means[candidate_indices])
    rows = []
    for i, record in enumerate(records):
        if i == 0:
            continue
        changes = []
        for chain, seq, parent_seq in (
            ("H", record.heavy_sequence, dataset.parent.heavy_sequence),
            ("L", record.light_sequence, dataset.parent.light_sequence),
        ):
            if seq:
                changes.extend(mutations(parent_seq, seq, chain, annotations[chain]))
        reasons = []
        for ok, label in (
            (unmeasured[i], "already measured"),
            (stability_ok[i], "stability confidence bound"),
            (charge_ok[i], "chain charge"),
            (hydro_ok[i], "hydrophobic patch"),
            (apr_ok[i], "aggregation proxy"),
        ):
            if not ok:
                reasons.append(label)
        rank = selected.index(i) + 1 if i in selected else None
        observation = observations.get(record.variant_id)
        rows.append(
            {
                **record.model_dump(),
                "binding_mean": float(means[i, 0]),
                "binding_std": float(sigmas[i, 0]),
                "stability_mean": float(means[i, 1]),
                "stability_std": float(sigmas[i, 1]),
                "raw_stability_mean": float(raw_stability[i]),
                "stability_lcb": float(lcb[i]),
                "developability_penalty": float(penalties[i]),
                "pseudo_perplexity": perplexities[i],
                "ehvi": float(ehvi[i]),
                "acquisition": float(acquisition[i]),
                "eligible": bool(eligible[i]),
                "exclusions": reasons,
                "pareto": bool(pareto[i]),
                "selected": rank is not None,
                "selection_rank": rank,
                "well": f"{'AB'[(rank - 1) // 12]}{(rank - 1) % 12 + 1:02d}" if rank else None,
                "measured": observation is not None,
                "observation": observation.model_dump() if observation else None,
                "mutations": changes,
                **properties[i],
            }
        )
    chosen_embeddings = embeddings[selected]
    distance = np.sqrt(
        np.maximum(
            np.sum(chosen_embeddings**2, axis=1)[:, None]
            + np.sum(chosen_embeddings**2, axis=1)[None, :]
            - 2 * chosen_embeddings @ chosen_embeddings.T,
            0,
        )
    )
    warnings = (
        ["Synthetic assay values: demonstration only."] if dataset.name.endswith("demo library") else []
    )
    warnings += [
        "CDRs are approximate reference-aligned regions, not standardized antibody numbering.",
        "Sequence liabilities and stability predictions do not establish structural preservation.",
    ]
    if config.embedding == "descriptor":
        warnings.append(
            "Offline descriptors are not pretrained protein embeddings; natural sequence likelihood is unavailable."
        )
    elif config.embedding == "ablang2":
        warnings.append("AbLang-2 supplies antibody embeddings; masked sequence likelihood is not computed in this mode.")
    return {
        "variants": rows,
        "selected_ids": [records[i].variant_id for i in selected],
        "parent": dataset.parent.model_dump(),
        "annotations": list(annotations.values()),
        "config": config.model_dump(),
        "summary": {
            "library_size": len(dataset.variants),
            "observed_variants": sum(r.variant_id in observations for r in dataset.variants),
            "training_points": len(training),
            "total_observations": len(observations),
            "eligible": int(eligible.sum()),
            "pareto_count": int(pareto.sum()),
            "selected": len(selected),
            "mean_binding": float(means[selected, 0].mean()),
            "mean_stability": float(means[selected, 1].mean()),
            "min_stability_lcb": float(lcb[selected].min()),
            "parent_binding": float(means[0, 0]),
            "mean_pairwise_distance": float(distance[np.triu_indices(24, 1)].mean()),
            "exploration_weight": exploration_weight,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        },
        "provenance": {
            "algorithm": "correlated-output marginal MC-EHVI (256 samples) + quality-weighted greedy DPP",
            "surrogate": "joint exact linear+RBF two-output GP; learned binding-stability covariance; latent epistemic standard deviations",
            "embedding": ESMEmbedder.model_name
            if config.embedding == "esm2"
            else AbLang2Embedder.model_name
            if config.embedding == "ablang2"
            else "deterministic positional descriptors v1 (not a pLM)",
            "seed": config.seed,
            "esm_revision": ESMEmbedder.revision if config.embedding == "esm2" else None,
            "dataset_sha256": hashlib.sha256(dataset.model_dump_json().encode()).hexdigest(),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "aerofold": "1.0.0",
            },
            "score_direction": "both maximized; stability is an adjusted 0–100 assay score",
            "training_ids": [records[i].variant_id for i in training],
            "training_observations": [observations[records[i].variant_id].model_dump() for i in training],
        },
        "warnings": warnings,
    }
