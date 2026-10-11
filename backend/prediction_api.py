"""Authenticated, bounded-job API for supervised predictions and benchmarks."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from backend.foundation_benchmarks import run_benchmark_suite
from backend.predictions import (
    PredictionBenchmarkConfig,
    PredictionConfig,
    partition,
    run_prediction,
)
from backend.public_data import catalog, load_task, parse_sequences, save_custom


def prediction_router(root: Path, store, executor, slots, authorize) -> APIRouter:
    """Share the application's concurrency limit and token policy."""
    router = APIRouter(prefix="/api/prediction", dependencies=[Depends(authorize)])
    logger = logging.getLogger("aerofold.prediction")

    def execute(identifier: str, config, kind: str) -> None:
        try:

            def progress(message: str) -> None:
                store.update_prediction_job(identifier, "running", message)

            if kind == "prediction_benchmark":
                result = run_benchmark_suite(
                    config,
                    root,
                    config.predictors,
                    progress,
                    config.embedding_batch_size,
                    config.local_only,
                )
            else:
                result = run_prediction(config, root, progress)
            store.update_prediction_job(identifier, "completed", "Complete", result=result)
        except Exception as exc:
            logger.exception("Prediction job failed")
            message = (
                str(exc)
                if isinstance(exc, (ValueError, ImportError))
                else "Prediction job failed; see the server log for details"
            )
            store.update_prediction_job(identifier, "failed", "Failed", error=message)
        finally:
            slots.release()

    def enqueue(config, kind: str) -> dict:
        metadata, records = load_task(config.task_id, root / "custom-tasks")
        if not metadata["ready"]:
            raise ValueError("This assay has too few examples for validated model training")
        if kind == "prediction_benchmark" or config.queries is None:
            partition(records, config.seed, getattr(config, "split", "random"))
        if 0 < config.train_limit < 60:
            raise ValueError("Choose at least 60 training sequences, or 0 to use all available")
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "A model job is running and another is queued; retry after completion")
        identifier = None
        try:
            identifier = store.create_prediction_job(kind, config.model_dump())
            executor.submit(execute, identifier, config, kind)
        except Exception:
            slots.release()
            if identifier:
                store.update_prediction_job(
                    identifier, "failed", "Could not schedule job", error="Executor unavailable"
                )
            raise
        return store.prediction_job(identifier)

    @router.get("/tasks")
    def tasks() -> dict:
        return catalog(root / "custom-tasks")

    @router.post("/tasks", status_code=201)
    async def upload_task(
        file: UploadFile = File(...),
        name: str = Form(..., min_length=1, max_length=120),
        target: str = Form(..., min_length=1, max_length=200),
        unit: str = Form(..., min_length=1, max_length=80),
        maximize: bool = Form(True),
        category: str = Form("binding", pattern="^(binding|expression|thermostability|other)$"),
    ) -> dict:
        raw = await file.read(5 * 1024 * 1024 + 1)
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(413, "Maximum upload size is 5 MB")
        return save_custom(root / "custom-tasks", raw, name, target, unit, maximize, category)

    @router.post("/run", status_code=202)
    def predict(config: PredictionConfig) -> dict:
        return enqueue(config, "prediction")

    @router.post("/upload", status_code=202)
    async def predict_uploaded(file: UploadFile = File(...), config: str = Form(...)) -> dict:
        raw = await file.read(5 * 1024 * 1024 + 1)
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(413, "Maximum upload size is 5 MB")
        settings = json.loads(config)
        settings["queries"] = parse_sequences(raw)
        return enqueue(PredictionConfig.model_validate(settings), "prediction")

    @router.post("/benchmark/run", status_code=202)
    def benchmark(config: PredictionBenchmarkConfig) -> dict:
        return enqueue(config, "prediction_benchmark")

    @router.get("/jobs")
    def history() -> list[dict]:
        return store.prediction_jobs()

    @router.get("/jobs/{identifier}")
    def job(identifier: str) -> dict:
        return store.prediction_job(identifier)

    @router.get("/jobs/{identifier}/export")
    def export(identifier: str, format: str = "csv") -> Response:
        job = store.prediction_job(identifier)
        if job["status"] != "completed":
            raise HTTPException(409, "Wait for the model job to finish")
        result = job["result"]
        if format == "json":
            return Response(
                json.dumps(result, indent=2, allow_nan=False),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="aerofold-{identifier}.json"'},
            )
        if format != "csv" or job["kind"] != "prediction":
            raise ValueError("Use format=json for benchmarks, or csv for prediction rankings")
        output = io.StringIO(newline="")
        fields = [
            "rank",
            "variant_id",
            "heavy_sequence",
            "light_sequence",
            "prediction",
            "lower",
            "upper",
            "model_disagreement",
            "nearest_training_identity",
            "support",
            "used_as_labeled_data",
            "observed_score",
            "assay",
            "unit",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in result["variants"]:
            entry = {**row, "assay": result["task"]["name"], "unit": result["task"]["unit"]}
            # Metadata can be user-supplied; neutralize spreadsheet formulas.
            writer.writerow(
                {
                    key: (
                        "'" + value
                        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r"))
                        else value
                    )
                    for key, value in entry.items()
                }
            )
        return Response(
            output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="aerofold-{identifier}.csv"'},
        )

    return router
