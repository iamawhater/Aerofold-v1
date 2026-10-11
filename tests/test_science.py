import numpy as np
import pytest

from backend.features import descriptor_embeddings, developability, fragment_mutation_embeddings
from backend.models import SelectionConfig
from backend.optimization import expected_hv_improvement, hypervolume, pareto_mask, select_diverse
from backend.pipeline import run_selection
from backend.sequences import DEMO_HEAVY, align, annotate, mutations
from backend.surrogate import CorrelatedGP, ExactGP


def test_hypervolume_and_ties():
    points = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    assert pareto_mask(points).tolist() == [True, True, True, False, True]
    assert hypervolume(points, np.zeros(2)) == pytest.approx(6.0)
    assert hypervolume(np.array([[-1.0, 2.0]]), np.zeros(2)) == 0


def test_ehvi_zero_uncertainty_equals_exact_hv_gain():
    baseline = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    candidates = np.array([[1.0, 1.0], [4.0, 4.0], [2.5, 2.5], [0.0, 5.0]])
    got = expected_hv_improvement(candidates, np.zeros_like(candidates), baseline, seed=4)
    norm = (baseline - baseline.min(0)) / np.ptp(baseline, axis=0)
    ref = np.array([-0.15, -0.15])
    expected = [
        hypervolume(np.vstack([norm, (p - baseline.min(0)) / np.ptp(baseline, axis=0)]), ref)
        - hypervolume(norm, ref)
        for p in candidates
    ]
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_dpp_exact_unique_deterministic_and_strict():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 8))
    quality = rng.uniform(size=80)
    eligible = np.arange(80) >= 15
    means = rng.normal(size=(80, 2))
    args = (x, quality, eligible, means, np.ones_like(means), 0.7)
    selected = select_diverse(*args)
    assert len(set(selected)) == 24
    assert all(eligible[selected])
    assert selected == select_diverse(*args)
    eligible[:60] = False
    with pytest.raises(ValueError, match="Only 20"):
        select_diverse(*args)


def test_dpp_handles_identical_embeddings():
    indices = select_diverse(
        np.ones((30, 8)), np.zeros(30), np.ones(30, dtype=bool), np.ones((30, 2)), np.ones((30, 2)), 1
    )
    assert len(set(indices)) == 24


def test_gp_epistemic_uncertainty_increases_off_training_data():
    x = np.linspace(-1, 1, 15)[:, None]
    gp = ExactGP().fit(x, np.sin(x[:, 0] * 2), steps=80)
    mean, sigma = gp.predict(np.array([[0.0], [20.0]]))
    assert abs(mean[0]) < 0.1
    assert sigma[1] > sigma[0] * 3
    assert np.all(sigma > 0)


def test_joint_gp_learns_objective_covariance():
    x = np.linspace(-1, 1, 18)[:, None]
    binding = np.sin(2 * x[:, 0])
    outcomes = np.column_stack((binding, 2 * binding + 0.1 * x[:, 0]))
    means, sigmas, covariance = CorrelatedGP().fit(x, outcomes, steps=65).predict(x[:3])
    assert means.shape == sigmas.shape == (3, 2)
    assert covariance.shape == (3, 2, 2)
    assert np.all(covariance[:, 0, 1] > 0)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-8)


def test_fragment_features_keep_mutation_identity_and_burden():
    x = fragment_mutation_embeddings(["AAAA", "CAAA", "AACA"], "AAAA")
    assert x.shape == (3, 82)
    assert np.count_nonzero(x[0]) == 0
    assert np.count_nonzero(x[1]) == np.count_nonzero(x[2]) == 3
    assert x[1, -2:].tolist() == [2.0, 0.0]
    assert x[2, -2:].tolist() == [0.0, 2.0]


def test_ehvi_uses_cross_objective_covariance():
    mean = np.array([[0.9, 0.9]])
    sigma = np.array([[0.5, 0.5]])
    baseline = np.array([[0.0, 0.0], [1.0, 1.0]])
    positive = np.array([[[0.25, 0.2], [0.2, 0.25]]])
    negative = np.array([[[0.25, -0.2], [-0.2, 0.25]]])
    gain_positive = expected_hv_improvement(mean, sigma, baseline, 4, covariance=positive)
    gain_negative = expected_hv_improvement(mean, sigma, baseline, 4, covariance=negative)
    assert gain_positive[0] > gain_negative[0]


def test_cdr_mapping_preserves_indels():
    annotation = annotate(DEMO_HEAVY, "H")
    assert len(annotation["regions"]) == 3
    inserted = DEMO_HEAVY[:101] + "A" + DEMO_HEAVY[101:]
    a, b = align(DEMO_HEAVY, inserted)
    assert a.replace("-", "") == DEMO_HEAVY
    assert b.replace("-", "") == inserted
    changes = mutations(DEMO_HEAVY, inserted, "H", annotation)
    assert len(changes) == 1 and changes[0]["from"] == "-"
    assert changes[0]["insertion"] == 1
    with pytest.raises(ValueError, match="identity"):
        annotate("A" * 100, "H")


def test_demo_unique_representations_and_chain_liabilities(demo):
    assert len(demo.variants) == 500
    assert len(demo.observations) == 49
    rows = [demo.parent, *demo.variants[:10]]
    embeddings = descriptor_embeddings(rows)
    assert embeddings.shape == (11, 102)
    assert np.isfinite(embeddings).all()
    assert np.linalg.norm(embeddings[1] - embeddings[0]) > 0
    props = developability(demo.parent)
    assert 0 <= props["apr"] <= 1
    assert props["max_chain_abs_charge"] >= abs(props["net_charge"]) / 2


def test_end_to_end_constraints_no_oracle_and_learning_round(demo, tmp_path):
    config = SelectionConfig()
    result = run_selection(demo, config, tmp_path)
    selected = [v for v in result["variants"] if v["selected"]]
    assert len(selected) == 24
    assert len(set(result["selected_ids"])) == 24
    assert {v["well"] for v in selected} == {f"{row}{col:02d}" for row in "AB" for col in range(1, 13)}
    assert all(v["stability_lcb"] >= config.min_stability for v in selected)
    assert not set(result["selected_ids"]) & {o.variant_id for o in demo.observations}
    assert result["provenance"]["training_ids"] == [
        r.variant_id
        for r in [demo.parent, *demo.variants]
        if r.variant_id in {o.variant_id for o in demo.observations}
    ]
    assert all(v["pseudo_perplexity"] is None for v in result["variants"])
    assert result["summary"]["training_points"] == 49
    from backend.demo import make_demo

    _, oracle = make_demo()
    updated = demo.model_copy(deep=True)
    updated.observations += [o for o in oracle if o.variant_id in result["selected_ids"]]
    next_result = run_selection(updated, config, tmp_path)
    assert not set(next_result["selected_ids"]) & set(result["selected_ids"])
    assert next_result["summary"]["training_points"] == 73


def test_insufficient_training_and_impossible_constraint(demo, tmp_path):
    sparse = demo.model_copy(deep=True)
    sparse.observations = []
    with pytest.raises(ValueError, match="At least 6"):
        run_selection(sparse, SelectionConfig(), tmp_path)
    with pytest.raises(ValueError, match="Only 0"):
        run_selection(demo, SelectionConfig(min_stability=100), tmp_path)
