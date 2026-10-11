"""FastAPI boundary, bounded background execution and validated CSV export."""

from __future__ import annotations

import csv
import hmac
import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.benchmarks import (
    BenchmarkConfig,
    run_benchmark,
    validate_budget,
)
from backend.benchmarks import (
    list_datasets as benchmark_datasets,
)
from backend.benchmarks import (
    load_dataset as load_benchmark,
)
from backend.demo import make_demo
from backend.middleware import RequestSizeLimitMiddleware
from backend.models import DatasetInput, Observation, ObservationBatch, SelectionConfig
from backend.pipeline import run_selection
from backend.prediction_api import prediction_router
from backend.sequences import annotate, parse_library
from backend.storage import Store

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("aerofold")
MAX_UPLOAD = 5 * 1024 * 1024


def create_app(data_dir: Path | None = None) -> FastAPI:
    """Application factory; use one API worker with this embedded job executor."""
    root = data_dir or Path(os.getenv("AEROFOLD_DATA_DIR", str(ROOT / "runtime")))
    store = Store(root / "aerofold.sqlite3")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aerofold")
    slots = threading.BoundedSemaphore(2)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        store.recover()
        yield
        executor.shutdown(wait=True)

    app = FastAPI(title="AeroFold-Ab", version="1.0.0", lifespan=lifespan)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_UPLOAD + 65536)
    app.state.store = store

    async def authorize(request: Request) -> None:
        token = os.getenv("AEROFOLD_API_TOKEN")
        if token and not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {token}"):
            raise HTTPException(401, "Valid bearer token required")

    @app.exception_handler(ValueError)
    async def invalid_input(_: Request, exc: ValueError) -> JSONResponse:
        message = (
            "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            if isinstance(exc, ValidationError)
            else str(exc)
        )
        return JSONResponse(status_code=422, content={"detail": message})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Error inputs can themselves contain NaN/Infinity; never echo them into JSON.
        message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422, content={"detail": message})

    @app.exception_handler(KeyError)
    async def missing(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})

    @app.middleware("http")
    async def headers_and_limits(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > MAX_UPLOAD + 65536):
            return JSONResponse(status_code=413, content={"detail": "Maximum request size is 5 MB"})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def dataset_summary(entry: dict) -> dict:
        dataset = DatasetInput.model_validate(entry["data"])
        return {
            "id": entry["id"],
            "name": entry["name"],
            "revision": entry["revision"],
            "demo": entry["demo"],
            "parent": dataset.parent.model_dump(),
            "variant_count": len(dataset.variants),
            "observation_count": len(dataset.observations),
            "annotations": [
                annotate(seq, chain)
                for chain, seq in (("H", dataset.parent.heavy_sequence), ("L", dataset.parent.light_sequence))
                if seq
            ],
            "runs": store.runs(entry["id"]),
        }

    def validate_parent(dataset: DatasetInput) -> None:
        for chain, sequence in (("H", dataset.parent.heavy_sequence), ("L", dataset.parent.light_sequence)):
            if sequence:
                annotate(sequence, chain)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": "1.0.0", "device": "cuda" if torch.cuda.is_available() else "cpu"}

    @app.get("/api/datasets", dependencies=[Depends(authorize)])
    def datasets() -> list[dict]:
        return store.list_datasets()

    @app.get("/api/benchmark/datasets", dependencies=[Depends(authorize)])
    def available_benchmarks() -> list[dict]:
        return benchmark_datasets()

    def execute_benchmark(identifier: str, config: BenchmarkConfig) -> None:
        try:
            result = run_benchmark(
                config,
                lambda partial: store.update_benchmark(
                    identifier, "running", partial["progress"], result=partial
                ),
            )
            store.update_benchmark(identifier, "completed", result["progress"], result=result)
        except ValueError as exc:
            store.update_benchmark(identifier, "failed", error=str(exc))
        except Exception:
            logger.exception("Benchmark %s failed", identifier)
            store.update_benchmark(identifier, "failed", error="Benchmark failed. Inspect the server logs.")
        finally:
            slots.release()

    @app.post("/api/benchmark/run", dependencies=[Depends(authorize)], status_code=202)
    def start_benchmark(config: BenchmarkConfig) -> dict:
        validate_budget(config, len(load_benchmark(config.dataset_id).ids))
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "The compute queue is full; wait for the current job")
        try:
            identifier = store.create_benchmark(config.model_dump())
            executor.submit(execute_benchmark, identifier, config)
        except Exception:
            slots.release()
            raise
        return {**store.benchmark(identifier), "poll_url": f"/api/benchmark/runs/{identifier}"}

    @app.get("/api/benchmark/runs/{identifier}", dependencies=[Depends(authorize)])
    def get_benchmark(identifier: str) -> dict:
        return store.benchmark(identifier)

    @app.post("/api/datasets/demo", dependencies=[Depends(authorize)], status_code=201)
    def demo() -> dict:
        dataset, _ = make_demo()
        return dataset_summary(store.create_dataset(dataset, demo=True))

    @app.post("/api/datasets", dependencies=[Depends(authorize)], status_code=201)
    def create_dataset(dataset: DatasetInput) -> dict:
        validate_parent(dataset)
        return dataset_summary(store.create_dataset(dataset))

    @app.post("/api/datasets/upload", dependencies=[Depends(authorize)], status_code=201)
    async def upload(file: UploadFile = File(...)) -> dict:
        content = await file.read(MAX_UPLOAD + 1)
        await file.close()
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, "Maximum upload size is 5 MB")
        dataset = parse_library(
            content, file.filename or "library.csv", Path(file.filename or "Library").stem[:100]
        )
        validate_parent(dataset)
        return dataset_summary(store.create_dataset(dataset))

    @app.get("/api/datasets/{dataset_id}", dependencies=[Depends(authorize)])
    def get_dataset(dataset_id: str) -> dict:
        return dataset_summary(store.dataset(dataset_id))

    @app.post("/api/datasets/{dataset_id}/observations", dependencies=[Depends(authorize)])
    def add_observations(dataset_id: str, batch: ObservationBatch) -> dict:
        return dataset_summary(store.add_observations(dataset_id, batch.observations))

    @app.post("/api/datasets/{dataset_id}/observations/upload", dependencies=[Depends(authorize)])
    async def upload_observations(dataset_id: str, file: UploadFile = File(...)) -> dict:
        content = await file.read(MAX_UPLOAD + 1)
        await file.close()
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, "Maximum upload size is 5 MB")
        try:
            reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
            if not reader.fieldnames or not {"variant_id", "binding", "stability"} <= set(reader.fieldnames):
                raise ValueError("Results CSV requires variant_id,binding,stability")
            results = ObservationBatch(
                observations=[
                    Observation(
                        variant_id=row["variant_id"], binding=row["binding"], stability=row["stability"]
                    )
                    for row in reader
                ]
            )
        except UnicodeDecodeError as exc:
            raise ValueError("Results CSV must be UTF-8") from exc
        return dataset_summary(store.add_observations(dataset_id, results.observations))

    def execute(run_id: str, dataset: DatasetInput, config: SelectionConfig) -> None:
        try:
            result = run_selection(
                dataset,
                config,
                root / "embeddings",
                lambda message: store.update_run(run_id, "running", message),
            )
            store.update_run(run_id, "completed", "Selection complete", result=result)
        except ValueError as exc:
            store.update_run(run_id, "failed", error=str(exc))
        except Exception:
            logger.exception("Selection %s failed", run_id)
            store.update_run(
                run_id,
                "failed",
                error="Selection failed. Inspect server logs; for ESM-2 verify optional dependencies and model availability.",
            )
        finally:
            slots.release()

    @app.post("/api/datasets/{dataset_id}/runs", dependencies=[Depends(authorize)], status_code=202)
    def select(dataset_id: str, config: SelectionConfig) -> dict:
        entry = store.dataset(dataset_id)
        dataset = DatasetInput.model_validate(entry["data"])
        if len(dataset.observations) < 6:
            raise ValueError("Upload at least 6 paired observations before running selection")
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "The selection queue is full; wait for the current run")
        try:
            run_id = store.create_run(dataset_id, entry["revision"], config.model_dump())
            executor.submit(execute, run_id, dataset, config)
        except Exception:
            slots.release()
            raise
        return {"id": run_id, "status": "queued"}

    @app.get("/api/runs/{run_id}", dependencies=[Depends(authorize)])
    def get_run(run_id: str) -> dict:
        return store.run(run_id)

    def completed_run(run_id: str) -> dict:
        run = store.run(run_id)
        if run["status"] != "completed":
            raise HTTPException(409, "The selection is not complete")
        if run["stale"]:
            raise HTTPException(
                409,
                "Measurements changed after this run; select a fresh batch before exporting or simulating",
            )
        return run

    @app.post("/api/runs/{run_id}/simulate", dependencies=[Depends(authorize)])
    def simulate(run_id: str) -> dict:
        run = completed_run(run_id)
        entry = store.dataset(run["dataset_id"])
        if not entry["demo"]:
            raise HTTPException(403, "Synthetic results are available only for demo datasets")
        _, oracle = make_demo()
        selected = set(run["result"]["selected_ids"])
        observations = [o for o in oracle if o.variant_id in selected]
        return dataset_summary(
            store.add_observations(run["dataset_id"], observations, expected_revision=run["revision"])
        )

    @app.get("/api/runs/{run_id}/export", dependencies=[Depends(authorize)])
    def export(run_id: str, format: str = "robot") -> Response:
        run = completed_run(run_id)
        if format not in {"robot", "synthesis", "results", "audit"}:
            raise ValueError("Export format must be robot, synthesis, results or audit")
        if format == "audit":
            return JSONResponse(
                run,
                headers={"Content-Disposition": f'attachment; filename="aerofold-{run_id[:8]}-audit.json"'},
            )
        selected = sorted(
            (r for r in run["result"]["variants"] if r["selected"]), key=lambda r: r["selection_rank"]
        )
        if len(selected) != 24:
            raise HTTPException(409, "Run does not contain exactly 24 selections")
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        if format == "results":
            writer.writerow(["variant_id", "binding", "stability"])
            writer.writerows([r["variant_id"], "", ""] for r in selected)
        elif format == "synthesis":
            writer.writerow(
                ["order_name", "plate_id", "well", "chain", "amino_acid_sequence", "sequence_type", "run_id"]
            )
            for r in selected:
                for chain, key in (("H", "heavy_sequence"), ("L", "light_sequence")):
                    if r[key]:
                        writer.writerow(
                            [
                                f"{r['variant_id']}-{chain}",
                                f"AEROFOLD-{chain}",
                                r["well"],
                                chain,
                                r[key],
                                "protein",
                                run_id,
                            ]
                        )
        else:
            writer.writerow(
                [
                    "plate_id",
                    "well",
                    "variant_id",
                    "heavy_sequence",
                    "light_sequence",
                    "binding_mean",
                    "binding_std",
                    "stability_mean",
                    "stability_std",
                    "run_id",
                ]
            )
            for r in selected:
                writer.writerow(
                    [
                        "AEROFOLD-01",
                        r["well"],
                        r["variant_id"],
                        r["heavy_sequence"],
                        r["light_sequence"],
                        *[
                            round(r[k], 5)
                            for k in ("binding_mean", "binding_std", "stability_mean", "stability_std")
                        ],
                        run_id,
                    ]
                )
        return Response(
            output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="aerofold-{run_id[:8]}-{format}.csv"'},
        )

    app.include_router(prediction_router(root, store, executor, slots, authorize))

    # Build once, then FastAPI serves the frontend and API from one origin.
    if (ROOT / "frontend" / "dist").exists():
        app.mount("/", StaticFiles(directory=ROOT / "frontend" / "dist", html=True), name="frontend")
    return app


app = create_app()
