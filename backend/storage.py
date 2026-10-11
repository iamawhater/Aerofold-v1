"""Transactional SQLite persistence for immutable experiment snapshots and results."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.models import DatasetInput, Observation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, data TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 1, demo INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id),
                revision INTEGER NOT NULL, status TEXT NOT NULL, progress TEXT NOT NULL,
                config TEXT NOT NULL, result TEXT, error TEXT, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS runs_dataset ON runs(dataset_id, created_at);
            CREATE TABLE IF NOT EXISTS benchmarks (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, progress TEXT NOT NULL,
                config TEXT NOT NULL, result TEXT, error TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prediction_jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                progress TEXT NOT NULL, config TEXT NOT NULL, result TEXT,
                error TEXT, created_at TEXT NOT NULL
            );
            """)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with sqlite3.connect(self.path, timeout=30) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            yield db

    def recover(self) -> None:
        """A stopped process cannot continue its in-memory jobs."""
        with self.connection() as db:
            db.execute(
                "UPDATE prediction_jobs SET status='failed', error='Server restarted; start a new prediction job.' WHERE status IN ('queued', 'running')"
            )
            db.execute(
                "UPDATE benchmarks SET status='failed', error='Server restarted during this simulation. Run a new benchmark.' WHERE status IN ('queued', 'running')"
            )
            db.execute(
                "UPDATE runs SET status='failed', error='Server restarted during this run. Start a new selection.' WHERE status IN ('queued', 'running')"
            )

    def create_prediction_job(self, kind: str, config: dict) -> str:
        """Persist supervised jobs separately from legacy selection runs."""
        identifier = uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                "INSERT INTO prediction_jobs VALUES (?, ?, 'queued', 'Queued', ?, NULL, NULL, ?)",
                (identifier, kind, json.dumps(config, allow_nan=False), utc_now()),
            )
        return identifier

    def update_prediction_job(
        self,
        identifier: str,
        status: str,
        progress: str = "",
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        with self.connection() as db:
            db.execute(
                "UPDATE prediction_jobs SET status=?, progress=?, result=COALESCE(?,result), error=? WHERE id=?",
                (
                    status,
                    progress,
                    json.dumps(result, allow_nan=False) if result is not None else None,
                    error,
                    identifier,
                ),
            )

    def prediction_job(self, identifier: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM prediction_jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError("Prediction job not found")
        result = dict(row)
        for key in ("config", "result"):
            result[key] = json.loads(result[key]) if result[key] else None
        return result

    def prediction_jobs(self) -> list[dict]:
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,kind,status,progress,error,created_at FROM prediction_jobs ORDER BY created_at DESC LIMIT 50"
                )
            ]

    def create_benchmark(self, config: dict) -> str:
        """Persist a benchmark independently of wet-lab datasets and revisions."""
        identifier = uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                "INSERT INTO benchmarks VALUES (?, 'queued', 'Queued', ?, NULL, NULL, ?)",
                (identifier, json.dumps(config), utc_now()),
            )
        return identifier

    def update_benchmark(
        self,
        identifier: str,
        status: str,
        progress: str = "",
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Atomically checkpoint complete rounds; preserve them if a worker fails."""
        with self.connection() as db:
            db.execute(
                "UPDATE benchmarks SET status=?,progress=?,result=COALESCE(?,result),error=? WHERE id=?",
                (
                    status,
                    progress,
                    json.dumps(result, allow_nan=False) if result is not None else None,
                    error,
                    identifier,
                ),
            )

    def benchmark(self, identifier: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM benchmarks WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError("Benchmark run not found")
        result = dict(row)
        for key in ("config", "result"):
            result[key] = json.loads(result[key]) if result[key] else None
        return result

    def create_dataset(self, dataset: DatasetInput, demo: bool = False) -> dict:
        identifier = uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, 1, ?, ?)",
                (identifier, dataset.name, dataset.model_dump_json(), int(demo), utc_now()),
            )
        return self.dataset(identifier)

    def dataset(self, identifier: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM datasets WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError("Dataset not found")
        result = dict(row)
        result["data"] = json.loads(result["data"])
        result["demo"] = bool(result["demo"])
        return result

    def list_datasets(self) -> list[dict]:
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,name,revision,demo,created_at FROM datasets ORDER BY created_at DESC"
                )
            ]

    def add_observations(
        self, identifier: str, observations: list[Observation], expected_revision: int | None = None
    ) -> dict:
        """Atomic append; measured values are immutable (no accidental overwrites)."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data,revision FROM datasets WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError("Dataset not found")
            if expected_revision is not None and row["revision"] != expected_revision:
                raise ValueError("This run is stale; select a new batch using the current dataset revision")
            dataset = DatasetInput.model_validate_json(row["data"])
            known = {r.variant_id for r in [dataset.parent, *dataset.variants]}
            previous = {o.variant_id for o in dataset.observations}
            ids = [o.variant_id for o in observations]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate result IDs")
            if set(ids) - known:
                raise ValueError(
                    "Result IDs not in this library: " + ", ".join(sorted(set(ids) - known)[:10])
                )
            if set(ids) & previous:
                raise ValueError(
                    "These variants already have measurements; result uploads never overwrite assays"
                )
            dataset.observations.extend(observations)
            db.execute(
                "UPDATE datasets SET data=?, revision=revision+1 WHERE id=?",
                (dataset.model_dump_json(), identifier),
            )
        return self.dataset(identifier)

    def create_run(self, dataset_id: str, revision: int, config: dict) -> str:
        identifier = uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, 'queued', 'Queued', ?, NULL, NULL, ?)",
                (identifier, dataset_id, revision, json.dumps(config), utc_now()),
            )
        return identifier

    def update_run(
        self,
        identifier: str,
        status: str,
        progress: str = "",
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        with self.connection() as db:
            db.execute(
                "UPDATE runs SET status=?,progress=?,result=?,error=? WHERE id=?",
                (
                    status,
                    progress,
                    json.dumps(result, allow_nan=False) if result is not None else None,
                    error,
                    identifier,
                ),
            )

    def run(self, identifier: str) -> dict:
        with self.connection() as db:
            row = db.execute(
                "SELECT runs.*, datasets.revision AS current_revision FROM runs JOIN datasets ON datasets.id=runs.dataset_id WHERE runs.id=?",
                (identifier,),
            ).fetchone()
        if row is None:
            raise KeyError("Run not found")
        result = dict(row)
        result["config"] = json.loads(result["config"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        result["stale"] = result["revision"] != result.pop("current_revision")
        return result

    def runs(self, dataset_id: str) -> list[dict]:
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,revision,status,created_at,error FROM runs WHERE dataset_id=? ORDER BY created_at DESC",
                    (dataset_id,),
                )
            ]
