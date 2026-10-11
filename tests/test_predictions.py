"""Regression protection for scientific splits, data semantics and supervised APIs."""

import csv
import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import predictions, public_data
from backend.main import create_app
from backend.prediction_models import PredictionEnsemble
from backend.predictions import PredictionBenchmarkConfig, PredictionConfig, metrics, partition
from scripts.sync_prediction_data import normalize


@pytest.fixture(scope="module")
def sample():
    metadata, records = public_data.load_task("titeseq_4420_binding")
    return metadata, records[:100]


def test_catalog_complete_and_units():
    cat = public_data.catalog()
    assert len(cat["tasks"]) == 17
    assert sum(t["unique_sequences"] for t in cat["tasks"]) == 74219
    for task in cat["tasks"]:
        metadata, records = public_data.load_task(task["id"])
        assert len(records) == len({public_data.fingerprint(r) for r in records})
        assert len(records) == metadata["unique_sequences"]
    _, rows = public_data.load_task("titeseq_4420_binding")
    assert 5 <= min(r["score"] for r in rows) <= max(r["score"] for r in rows) <= 9.51
    _, rows = public_data.load_task("igdesign_tslp_binding")
    assert 3 < min(r["score"] for r in rows) < max(r["score"] for r in rows) < 13


def test_source_transform_and_duplicate_grouping(sample):
    _, rows = sample
    row = rows[0]
    raw = io.StringIO()
    writer = csv.DictWriter(raw, fieldnames=["heavy", "light", "KD (nM)"])
    writer.writeheader()
    for score in [1, 100]:
        writer.writerow({"heavy": row["heavy_sequence"], "light": row["light_sequence"], "KD (nM)": score})
    parsed, qc = normalize({"column": "KD (nM)", "transform": "pkd_nm"}, raw.getvalue().encode())
    assert len(parsed) == 1 and parsed[0]["score"] == 8
    assert qc["duplicates_aggregated"] == 1


def test_checksum_and_path_safety(tmp_path, monkeypatch):
    (tmp_path / "catalog.json").write_text(
        json.dumps({"tasks": [{"id": "test", "file": "test.csv", "csv_sha256": "wrong"}]})
    )
    (tmp_path / "test.csv").write_text("tampered")
    monkeypatch.setattr(public_data, "PUBLIC", tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        public_data.load_task("test")
    with pytest.raises(KeyError):
        public_data.load_task("../test")


def test_split_score_independence_and_duplicates(sample):
    _, rows = sample
    development, test = partition(rows, 19)
    changed = [{**r, "score": -999 * i} for i, r in enumerate(rows)]
    other_dev, other_test = partition(changed, 19)
    assert np.array_equal(development, other_dev) and np.array_equal(test, other_test)
    assert not set(development) & set(test)
    with pytest.raises(ValueError, match="Duplicate"):
        partition(rows + [rows[0]], 19)


def test_metrics_direction_ties_and_calibration():
    truth = np.arange(100.0)
    best = metrics(truth, truth, True, 24, truth - 1, truth + 1)
    inverse = metrics(truth, truth, False, 24)
    assert best["top_recall"] == inverse["top_recall"] == 1
    assert best["top_enrichment"] == pytest.approx(100 / 24)
    assert best["coverage"] == 1 and best["rmse"] == 0
    tied = metrics(np.ones(100), np.arange(100.0), True, 24)
    assert tied["spearman"] is None
    assert tied["top_recall"] == 0.24 and tied["top_enrichment"] == 1


def test_calibration_labels_cannot_change_model_predictions(sample):
    _, rows = sample
    y = np.array([r["score"] for r in rows])
    a = PredictionEnsemble(seed=17).fit(rows, y)
    changed = y.copy()
    cal_ids = set(a.calibration_ids)
    changed[[i for i, r in enumerate(rows) if r["variant_id"] in cal_ids]] += 50
    b = PredictionEnsemble(seed=17).fit(rows, changed)
    assert a.validation == b.validation
    np.testing.assert_allclose(
        a.predict(rows, support=False)["prediction"], b.predict(rows, support=False)["prediction"]
    )
    assert b.radius > a.radius + 40
    assert sum(a.weights) == pytest.approx(1)
    assert np.min(a.weights) >= 0


def test_benchmark_never_fits_oracle_labels(sample, tmp_path, monkeypatch):
    metadata, rows = sample
    monkeypatch.setattr(predictions, "load_task", lambda *_: (metadata, rows))
    _, test = partition(rows, 42)
    hidden_ids = {rows[i]["variant_id"] for i in test}
    original_fit = PredictionEnsemble.fit
    calls = []

    def spy(self, data, scores, *args, **kwargs):
        assert not hidden_ids.intersection(r["variant_id"] for r in data)
        calls.append(len(data))
        return original_fit(self, data, scores, *args, **kwargs)

    monkeypatch.setattr(PredictionEnsemble, "fit", spy)
    result = predictions.run_prediction_benchmark(
        PredictionBenchmarkConfig(representation="sequence", train_limit=80, repeats=1, test_limit=20),
        tmp_path,
    )
    assert calls == [60, 80]
    assert result["split_audit"][0]["test_ids"]
    assert result["curves"][-1]["ensemble"]["coverage"] is not None


def test_prediction_demo_is_unseen_and_ranked(sample, tmp_path, monkeypatch):
    metadata, rows = sample
    monkeypatch.setattr(predictions, "load_task", lambda *_: (metadata, rows))
    result = predictions.run_prediction(PredictionConfig(representation="sequence", train_limit=80), tmp_path)
    assert len(result["variants"]) == 20
    assert not any(r["used_as_labeled_data"] for r in result["variants"])
    assert result["evaluation"]["test_count"] == 20
    assert all(r["lower"] <= r["prediction"] <= r["upper"] for r in result["variants"])
    assert [r["rank"] for r in result["variants"]] == list(range(1, 21))


def test_custom_import_api_auth_and_job_lifecycle(tmp_path, sample, monkeypatch):
    _, rows = sample
    content = io.StringIO()
    writer = csv.DictWriter(
        content, fieldnames=["variant_id", "heavy_sequence", "light_sequence", "score"], extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(rows)
    monkeypatch.setenv("AEROFOLD_API_TOKEN", "test-secret")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/prediction/tasks").status_code == 401
        client.headers["Authorization"] = "Bearer test-secret"
        response = client.post(
            "/api/prediction/tasks",
            files={"file": ("train.csv", content.getvalue())},
            data={
                "name": "=formula",
                "target": "Test antigen",
                "unit": "pKd",
                "maximize": "false",
                "category": "binding",
            },
        )
        assert response.status_code == 201, response.text
        task = response.json()
        assert task["maximize"] is False
        config = {"task_id": task["id"], "representation": "sequence", "train_limit": 80}
        response = client.post("/api/prediction/run", json=config)
        assert response.status_code == 202, response.text
        identifier = response.json()["id"]
        for _ in range(200):
            job = client.get(f"/api/prediction/jobs/{identifier}").json()
            if job["status"] not in {"running", "queued"}:
                break
            time.sleep(0.05)
        assert job["status"] == "completed", job
        data = client.get(f"/api/prediction/jobs/{identifier}/export").text
        assert "'=formula" in data
        exported = list(csv.DictReader(io.StringIO(data)))
        assert len(exported) == 20
        assert client.get(f"/api/prediction/jobs/{identifier}/export?format=json").status_code == 200
        assert client.get("/api/prediction/jobs").json()[0]["id"] == identifier
