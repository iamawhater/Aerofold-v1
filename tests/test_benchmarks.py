"""Benchmark fairness, metric identities, deterministic replay and job lifecycle."""

import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import benchmarks as bm
from backend.features import fragment_mutation_embeddings
from backend.main import create_app
from backend.storage import Store


@pytest.fixture(scope="module")
def paired_runs():
    config = bm.BenchmarkConfig(rounds=2)
    return bm.run_benchmark(config), bm.run_benchmark(config)


def test_public_data_and_checksum(tmp_path, monkeypatch):
    data = bm.load_dataset("titeseq_4_4_20")
    assert len(data.ids) == len(set(data.sequences)) == 512
    assert not data.oracle.flags.writeable
    assert "expression" in data.metadata["stability_label"].lower()
    assert data.metadata["source_commit"] == "54130c8fc6081bbfa8301c41e1afcfa6a5d79590"
    for path in bm.DATA_DIR.iterdir():
        if path.suffix in {".json", ".csv"}:
            (tmp_path / path.name).write_bytes(path.read_bytes())
    with (tmp_path / "titeseq_4_4_20.csv").open("a") as f:
        f.write("tampered")
    monkeypatch.setattr(bm, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        bm.load_dataset("titeseq_4_4_20")
    with pytest.raises(ValueError, match="Unknown"):
        bm.load_dataset("../../secret")


def test_metric_ties_and_undefined():
    assert bm.spearman(np.array([1, 1, 3, 4]), np.array([4, 4, 2, 1])) == pytest.approx(-1)
    assert bm.spearman(np.ones(4), np.arange(4)) is None
    assert bm.spearman(np.array([]), np.array([])) is None


def test_joint_benchmark_policy_respects_censored_binding_range():
    data = bm.load_dataset("titeseq_4_4_20")
    embeddings = fragment_mutation_embeddings(data.sequences, data.metadata["parent_sequence"])
    observed = np.arange(20)
    policy = bm.fit_policy(embeddings, observed, data.oracle[observed], bm.BenchmarkConfig())
    assert np.all((policy["means"][:, 0] >= 5) & (policy["means"][:, 0] <= 9.5))
    assert policy["covariances"].shape == (512, 2, 2)
    assert np.max(np.abs(policy["covariances"][:, 0, 1])) > 0


def test_reproducible_equal_budget_and_cumulative_metrics(paired_runs):
    first, second = paired_runs
    assert first["rounds"] == second["rounds"]
    assert first["status"] == "completed"
    assert first["rounds"][0]["active"] == first["rounds"][0]["random"]
    for agent in ("active", "random"):
        previous_recall, previous_hv, previous_ids = 0, 0, set()
        for r in first["rounds"]:
            m = r[agent]
            ids = set(m["observed_ids"])
            assert len(ids) == m["observed_count"] == 20 + 24 * r["round"]
            assert len(m["selected_ids"]) == (24 if r["round"] else 20)
            assert previous_ids <= ids
            assert previous_recall <= m["top_k_recall"] <= 100
            assert (
                previous_hv <= m["hypervolume"] <= first["protocol"]["hypervolume"]["oracle_maximum"] + 1e-12
            )
            assert not ids.intersection(r["evaluation_ids"])
            assert m["enrichment"] == pytest.approx(
                m["top_hits"] / m["observed_count"] / (first["top_k"]["actual_count_including_ties"] / 512)
            )
            previous_recall, previous_hv, previous_ids = m["top_k_recall"], m["hypervolume"], ids
    json.dumps(first, allow_nan=False)


def test_hidden_oracle_changes_cannot_change_first_batch(monkeypatch, paired_runs):
    original = bm.load_dataset("titeseq_4_4_20")
    baseline = paired_runs[0]
    labels = original.oracle.copy()
    unseen = np.array([identifier not in baseline["seed_ids"] for identifier in original.ids])
    labels[unseen, 0] = np.linspace(5, 9.5, unseen.sum())[::-1]
    labels[unseen, 1] = 100 - labels[unseen, 1]
    monkeypatch.setattr(
        bm,
        "load_dataset",
        lambda _: bm.BenchmarkData(original.metadata, original.ids, original.sequences, labels),
    )
    changed = bm.run_benchmark(bm.BenchmarkConfig(rounds=1))
    for agent in ("active", "random"):
        assert changed["rounds"][1][agent]["selected_ids"] == baseline["rounds"][1][agent]["selected_ids"]


def test_infeasible_stops_without_partial_or_relaxed_batch():
    result = bm.run_benchmark(bm.BenchmarkConfig(rounds=2, min_stability=100, confidence_beta=3))
    assert result["status"] == "stopped_early"
    assert len(result["rounds"]) == 1
    assert result["summary"]["assays_per_agent"] == 20
    assert "feasible" in result["stop_reason"]


def test_random_uses_only_unmeasured_feasible_candidates():
    eligible = np.zeros(100, dtype=bool)
    eligible[30:60] = True
    chosen = bm.choose_batch(
        {"eligible": eligible},
        np.zeros((100, 2)),
        np.zeros((20, 2)),
        bm.BenchmarkConfig(),
        np.random.default_rng(4),
        "random",
        1,
    )
    assert len(set(chosen)) == 24
    assert eligible[chosen].all()


def test_api_job_validation_auth_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("AEROFOLD_API_TOKEN", "benchmark-test")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/benchmark/datasets").status_code == 401
        client.headers["Authorization"] = "Bearer benchmark-test"
        assert client.get("/api/benchmark/datasets").json()[0]["variant_count"] == 512
        for body in (
            {"rounds": 12, "batch_size": 48},
            {"dataset_id": "../../bad"},
            {"batch_size": 23},
            {"exploration": 1.1},
        ):
            assert client.post("/api/benchmark/run", json=body).status_code == 422
        response = client.post("/api/benchmark/run", json={"rounds": 1, "batch_size": 12})
        assert response.status_code == 202
        url = response.json()["poll_url"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = client.get(url).json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        assert job["status"] == "completed", job
        assert job["result"]["summary"]["assays_per_agent"] == 32
        assert client.get("/api/benchmark/runs/missing").status_code == 404
    store = Store(tmp_path / "aerofold.sqlite3")
    identifier = store.create_benchmark({})
    store.update_benchmark(identifier, "running", result={"rounds": [0]})
    store.recover()
    assert store.benchmark(identifier)["status"] == "failed"
    assert store.benchmark(identifier)["result"] == {"rounds": [0]}


def test_bounded_shared_queue(tmp_path, monkeypatch):
    release = threading.Event()
    entered = threading.Event()

    def blocked(config, progress):
        entered.set()
        release.wait(timeout=10)
        return {"progress": "Done"}

    monkeypatch.setattr("backend.main.run_benchmark", blocked)
    with TestClient(create_app(tmp_path)) as client:
        try:
            assert client.post("/api/benchmark/run", json={}).status_code == 202
            assert entered.wait(timeout=5)
            assert client.post("/api/benchmark/run", json={}).status_code == 202
            assert client.post("/api/benchmark/run", json={}).status_code == 429
        finally:
            release.set()
