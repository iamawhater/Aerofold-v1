"""Standalone embedding comparators: pooling, split fairness and label isolation."""

import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend import foundation_benchmarks as fb
from backend import predictions
from backend.main import create_app
from backend.prediction_models import PredictionEnsemble
from backend.predictions import PredictionBenchmarkConfig, benchmark_plans
from backend.public_data import fingerprint, load_task


def test_special_tokens_excluded_from_pooling():
    tokens = torch.tensor([[999.0, 999.0], [1.0, 3.0], [3.0, 5.0], [999.0, 999.0]])
    np.testing.assert_array_equal(fb.pool_residues(tokens, "AC"), [2, 4])
    with pytest.raises(ValueError, match="alignment"):
        fb.pool_residues(tokens, "ACD")


def test_only_embedding_head_and_calibration_independence():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(100, 16))
    y = x[:, 0] * 2 + rng.normal(scale=0.1, size=100)
    ids = [f"seq-{i}" for i in range(100)]
    a = fb.EmbeddingOnlyRegressor(9, "esm2-only").fit(x, y, ids)
    changed = y.copy()
    changed[[ids.index(identifier) for identifier in a.calibration_ids]] += 100
    b = fb.EmbeddingOnlyRegressor(9, "esm2-only").fit(x, changed, ids)
    assert a.alpha == b.alpha and a.validation_rmse == b.validation_rmse
    np.testing.assert_allclose(a.predict(x)["prediction"], b.predict(x)["prediction"])
    assert b.radius > a.radius + 95
    assert not set(a.calibration_ids).intersection(a.fitting_ids)
    assert list(a.model.named_steps) == ["standardscaler", "ridge"]
    assert a.card()["head"].endswith("frozen embeddings only")


def test_suite_same_splits_and_no_test_labels_in_any_head(tmp_path, monkeypatch):
    metadata, records = load_task("titeseq_4420_binding")
    records = records[:100]
    monkeypatch.setattr(fb, "load_task", lambda *_: (metadata, records))
    monkeypatch.setattr(predictions, "load_task", lambda *_: (metadata, records))
    config = PredictionBenchmarkConfig(representation="sequence", train_limit=80, test_limit=20, repeats=1)
    _, development, test, _ = benchmark_plans(config, records)[0]
    hidden_ids = {records[i]["variant_id"] for i in test}
    encoded_calls, fit_calls = [], []

    def fake_embeddings(rows, predictor, *args):
        encoded_calls.append(predictor)
        # Fake label-free encoder used only for the integration test, never real reports.
        values = np.stack([np.random.default_rng(int(fingerprint(r)[:8], 16)).normal(size=8) for r in rows])
        return values, {"test_fixture": True, "predictor": predictor}

    monkeypatch.setattr(fb, "frozen_embeddings", fake_embeddings)
    original = fb.EmbeddingOnlyRegressor.fit

    def spy(self, embeddings, scores, ids):
        assert not hidden_ids.intersection(ids)
        fit_calls.append((self.predictor, len(ids)))
        return original(self, embeddings, scores, ids)

    monkeypatch.setattr(fb.EmbeddingOnlyRegressor, "fit", spy)
    result = fb.run_benchmark_suite(config, tmp_path, ("ensemble", "esm2-only", "antiberty-only"))
    assert encoded_calls == ["esm2-only", "antiberty-only"]
    assert fit_calls == [("esm2-only", 60), ("esm2-only", 80), ("antiberty-only", 60), ("antiberty-only", 80)]
    assert result["split_audit"][0]["training_ids"] == [records[i]["variant_id"] for i in development]
    for run in result["runs"]:
        ensemble_cal = run["model_card"]["calibration_ids"]
        for predictor in ("esm2-only", "antiberty-only"):
            assert run["predictor_cards"][predictor]["calibration_ids"] == ensemble_cal
            assert run[predictor]["test_count"] == run["ensemble"]["test_count"]
    assert all("esm2-only" in point and "antiberty-only" in point for point in result["curves"])


def test_standalone_does_not_fit_sequence_ensemble(tmp_path, monkeypatch):
    metadata, records = load_task("titeseq_4420_binding")
    records = records[:100]
    monkeypatch.setattr(fb, "load_task", lambda *_: (metadata, records))

    def forbidden(*args, **kwargs):
        pytest.fail("Standalone mode must not fit mutation or ensemble heads")

    monkeypatch.setattr(PredictionEnsemble, "fit", forbidden)
    monkeypatch.setattr(
        fb,
        "frozen_embeddings",
        lambda rows, *args: (
            np.array([[len(r["heavy_sequence"]), int(fingerprint(r)[:4], 16)] for r in rows], dtype=float),
            {"test_fixture": True},
        ),
    )
    config = PredictionBenchmarkConfig(train_limit=80, repeats=1, test_limit=20)
    result = fb.run_benchmark_suite(config, tmp_path, ("antiberty-only",))
    assert result["predictors"] == ["antiberty-only"]
    assert "ensemble" not in result["curves"][0]
    assert len(result["runs"]) == 2
    with pytest.raises(ValueError, match="unique predictors"):
        fb.run_benchmark_suite(config, tmp_path, ("esm2-only", "esm2-only"))


@pytest.mark.parametrize(
    "predictors",
    [
        ["esm2-only"],
        ["antiberty-only"],
        ["ensemble", "esm2-only", "antiberty-only"],
    ],
)
def test_benchmark_api_dispatch_persistence_and_export(tmp_path, monkeypatch, predictors):
    """The dashboard request reaches the suite and persists distinct model audits."""
    from backend import prediction_api

    metadata, records = load_task("titeseq_4420_binding")
    records = records[:100]
    for module in (fb, predictions, prediction_api):
        monkeypatch.setattr(module, "load_task", lambda *_: (metadata, records))
    monkeypatch.delenv("AEROFOLD_API_TOKEN", raising=False)
    encoder_calls = []

    def encode(rows, predictor, root, progress, batch_size, local_only):
        encoder_calls.append((predictor, batch_size, local_only))
        progress(f"Encoding {predictor}")
        return np.array([np.random.default_rng(int(fingerprint(r)[:8], 16)).normal(size=8) for r in rows]), {
            "test_fixture": True,
            "predictor": predictor,
        }

    monkeypatch.setattr(fb, "frozen_embeddings", encode)
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/prediction/benchmark/run",
            json={
                "representation": "sequence",
                "train_limit": 80,
                "test_limit": 20,
                "repeats": 1,
                "predictors": predictors,
                "embedding_batch_size": 8,
                "local_only": True,
            },
        )
        assert response.status_code == 202, response.text
        identifier = response.json()["id"]
        for _ in range(200):
            job = client.get(f"/api/prediction/jobs/{identifier}").json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.05)
        assert job["status"] == "completed", job
        result = job["result"]
        assert result["predictors"] == predictors
        assert encoder_calls == [(p, 8, True) for p in predictors if p != "ensemble"]
        assert result["config"]["predictors"] == predictors
        for point in result["curves"]:
            assert all(p in point for p in predictors)
            assert ("ensemble" in point) == ("ensemble" in predictors)
        exported = client.get(f"/api/prediction/jobs/{identifier}/export?format=json").json()
        assert exported == result
        assert client.get("/api/prediction/jobs").json()[0]["id"] == identifier


@pytest.mark.parametrize("predictors", [[], ["unknown"], ["esm2-only", "esm2-only"]])
def test_benchmark_api_rejects_invalid_predictors(tmp_path, monkeypatch, predictors):
    monkeypatch.delenv("AEROFOLD_API_TOKEN", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/prediction/benchmark/run", json={"predictors": predictors})
        assert response.status_code == 422
        assert client.get("/api/prediction/jobs").json() == []
