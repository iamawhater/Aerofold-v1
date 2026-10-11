import csv
import io
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.demo import library_csv
from backend.main import create_app
from backend.models import DatasetInput, Observation, SelectionConfig, SequenceRecord
from backend.sequences import parse_library
from backend.storage import Store


def test_csv_and_paired_fasta_round_trip(demo):
    parsed = parse_library(library_csv(demo).encode(), "library.csv")
    assert parsed.parent == demo.parent
    assert parsed.variants == demo.variants
    assert parsed.observations == demo.observations
    content = "\n".join(
        f">{r.variant_id}/{chain}\n{seq}"
        for r in [demo.parent, *demo.variants]
        for chain, seq in [("H", r.heavy_sequence), ("L", r.light_sequence)]
    )
    parsed = parse_library(content.encode(), "library.fasta")
    assert parsed.variants == demo.variants
    assert parsed.observations == []


@pytest.mark.parametrize("binding", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite(binding):
    with pytest.raises(ValidationError):
        Observation(variant_id="AB-001", binding=binding, stability=80)


def test_reject_duplicate_sequences_ids_and_missing_chains(demo):
    data = demo.model_dump()
    data["variants"][1]["heavy_sequence"] = data["variants"][0]["heavy_sequence"]
    data["variants"][1]["light_sequence"] = data["variants"][0]["light_sequence"]
    with pytest.raises(ValidationError, match="Duplicate sequence"):
        DatasetInput.model_validate(data)
    data = demo.model_dump()
    data["variants"][1]["variant_id"] = data["variants"][0]["variant_id"]
    with pytest.raises(ValidationError, match="IDs must be unique"):
        DatasetInput.model_validate(data)
    data = demo.model_dump()
    data["variants"][0]["light_sequence"] = ""
    with pytest.raises(ValidationError, match="same chain types"):
        DatasetInput.model_validate(data)


def test_input_rejections_and_fixed_batch():
    with pytest.raises(ValueError, match="headers"):
        parse_library(b">parent\nAAAA\n>parent\nAAAA", "x.fasta")
    with pytest.raises(ValueError, match="Supported"):
        parse_library(b"anything", "x.exe")
    with pytest.raises(ValueError, match="UTF-8"):
        parse_library(b"\xff\xfe", "x.csv")
    with pytest.raises(ValidationError):
        SequenceRecord(variant_id="=SUM(A1)", heavy_sequence="A" * 100)
    with pytest.raises(ValidationError):
        SelectionConfig(batch_size=23)
    with pytest.raises(ValidationError):
        Observation(variant_id="AB-0001", binding=7, stability=101)


def test_transactional_measurements_and_recovery(demo, tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    entry = store.create_dataset(demo)
    new_id = next(
        v.variant_id for v in demo.variants if v.variant_id not in {o.variant_id for o in demo.observations}
    )
    valid = Observation(variant_id=new_id, binding=8, stability=82)
    bad = Observation(variant_id="nonexistent", binding=8, stability=82)
    with pytest.raises(ValueError, match="not in"):
        store.add_observations(entry["id"], [valid, bad])
    assert store.dataset(entry["id"])["revision"] == 1
    assert store.add_observations(entry["id"], [valid])["revision"] == 2
    with pytest.raises(ValueError, match="already have"):
        store.add_observations(entry["id"], [valid])
    run = store.create_run(entry["id"], 1, SelectionConfig().model_dump())
    assert store.run(run)["stale"]
    store.recover()
    assert store.run(run)["status"] == "failed"


def wait_for_run(client, run_id):
    for _ in range(1200):
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in {"completed", "failed"}:
            return run
        time.sleep(0.05)
    pytest.fail("Run did not finish within sixty seconds")


def test_api_round_trip_exports_simulation_and_stale_protection(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        entry = client.post("/api/datasets/demo").json()
        assert entry["variant_count"] == 500
        assert entry["observation_count"] == 49
        job = client.post(f"/api/datasets/{entry['id']}/runs", json={})
        assert job.status_code == 202
        run_id = job.json()["id"]
        run = wait_for_run(client, run_id)
        assert run["status"] == "completed", run.get("error")
        export = client.get(f"/api/runs/{run_id}/export")
        assert export.status_code == 200
        rows = list(csv.DictReader(io.StringIO(export.text)))
        assert len(rows) == 24
        assert len({r["well"] for r in rows}) == 24
        assert all(r["heavy_sequence"] and r["light_sequence"] for r in rows)
        synthesis = client.get(f"/api/runs/{run_id}/export?format=synthesis")
        orders = list(csv.DictReader(io.StringIO(synthesis.text)))
        assert len(orders) == 48
        assert len({(r["plate_id"], r["well"]) for r in orders}) == 48
        assert client.get(f"/api/runs/{run_id}/export?format=invalid").status_code == 422
        assert client.get("/api/runs/missing").status_code == 404
        updated = client.post(f"/api/runs/{run_id}/simulate")
        assert updated.status_code == 200, updated.text
        assert updated.json()["observation_count"] == 73
        assert client.get(f"/api/runs/{run_id}").json()["stale"]
        assert client.get(f"/api/runs/{run_id}/export").status_code == 409
        assert client.post(f"/api/runs/{run_id}/simulate").status_code == 409


def test_api_upload_validation_and_token(tmp_path, demo, monkeypatch):
    monkeypatch.setenv("AEROFOLD_API_TOKEN", "test-secret")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/datasets").status_code == 401
        client.headers["Authorization"] = "Bearer test-secret"
        response = client.post(
            "/api/datasets/upload", files={"file": ("library.csv", library_csv(demo).encode(), "text/csv")}
        )
        assert response.status_code == 201, response.text
        entry = response.json()
        assert not entry["demo"]
        response = client.post(
            "/api/datasets/upload", files={"file": ("broken.csv", b"wrong,columns\n1,2", "text/csv")}
        )
        assert response.status_code == 422
        response = client.post(
            f"/api/datasets/{entry['id']}/observations/upload",
            files={"file": ("results.csv", b"variant_id,binding,stability\nUNKNOWN,8,80", "text/csv")},
        )
        assert response.status_code == 422
        response = client.post(
            "/api/datasets/upload", files={"file": ("large.csv", b"x" * (6 * 1024 * 1024), "text/csv")}
        )
        assert response.status_code == 413


def test_chunked_body_limit(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        chunks = (b"x" * (1024 * 1024) for _ in range(6))
        response = client.post("/api/datasets", content=chunks, headers={"Content-Type": "application/json"})
        assert "content-length" not in response.request.headers
        assert response.status_code == 413


def test_nonfinite_json_errors_remain_valid_json(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/datasets/missing/observations",
            content='{"observations":[{"variant_id":"AB-0001","binding":NaN,"stability":80}]}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "finite" in response.json()["detail"]
