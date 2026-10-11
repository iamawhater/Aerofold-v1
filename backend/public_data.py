"""Integrity-checked, task-specific experimental data for supervised prediction."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import median

from backend.models import SequenceRecord

PUBLIC = Path(__file__).resolve().parents[1] / "data/public"


def fingerprint(record: dict) -> str:
    """Identity is the ordered pair of complete variable domains, not a row ID."""
    return hashlib.sha256(
        (record["heavy_sequence"] + "|" + record.get("light_sequence", "")).encode()
    ).hexdigest()


def catalog(custom_root: Path | None = None) -> dict:
    """Return provenance, assay semantics, exclusions and usable task metadata."""
    result = json.loads((PUBLIC / "catalog.json").read_text(encoding="utf-8"))
    if custom_root and custom_root.exists():
        result["tasks"].extend(
            json.loads(p.read_text(encoding="utf-8"))["metadata"] for p in sorted(custom_root.glob("*.json"))
        )
    return result


def load_task(task_id: str, custom_root: Path | None = None) -> tuple[dict, list[dict]]:
    """Validate the manifest and CSV on every load; never accept arbitrary paths."""
    entry = next((t for t in catalog(custom_root)["tasks"] if t["id"] == task_id), None)
    if entry is None:
        raise KeyError("Unknown prediction task")
    if entry.get("custom"):
        payload = json.loads((custom_root / f"{task_id}.json").read_text(encoding="utf-8"))
        return entry, payload["records"]
    path = (PUBLIC / entry["file"]).resolve()
    if not path.is_relative_to(PUBLIC.resolve()):
        raise ValueError("Invalid task manifest path")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["csv_sha256"]:
        raise ValueError("Packaged assay checksum mismatch; restore or rebuild the source catalog")
    records = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    for row in records:
        row["score"] = float(row["score"])
    return entry, records


def parse_sequences(raw: bytes, measured: bool = False) -> list[dict]:
    """Strict CSV input. Assay scores must already use the declared native units."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    required = {"heavy_sequence"} | ({"score"} if measured else set())
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(
            "CSV requires heavy_sequence, optional light_sequence and variant_id"
            + (", and score" if measured else "")
        )
    records, ids = [], set()
    for i, row in enumerate(reader):
        if i >= 12000:
            raise ValueError("Upload limit is 12,000 rows")
        record = SequenceRecord(
            variant_id=row.get("variant_id") or f"variant-{i + 1}",
            heavy_sequence=row["heavy_sequence"].strip().upper(),
            light_sequence=(row.get("light_sequence") or "").strip().upper(),
        ).model_dump()
        if record["variant_id"] in ids:
            raise ValueError("Variant IDs must be unique")
        ids.add(record["variant_id"])
        if measured:
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError("All assay scores must be finite")
            record["score"] = score
        records.append(record)
    if not records:
        raise ValueError("CSV contains no sequences")
    return records


def save_custom(
    root: Path, raw: bytes, name: str, target: str, unit: str, maximize: bool, category: str
) -> dict:
    """Aggregate replicate sequences before any split; keep each uploaded assay separate."""
    records = parse_sequences(raw, measured=True)
    groups = defaultdict(list)
    for row in records:
        groups[fingerprint(row)].append(row)
    unique = [
        {
            **rows[0],
            "score": median(r["score"] for r in rows),
            "replicate_count": len(rows),
            "sequence_sha256": key,
        }
        for key, rows in groups.items()
    ]
    if len(unique) < 60:
        raise ValueError(
            "At least 60 unique sequence pairs are required for fitting, validation and calibration"
        )
    identifier = "custom_" + uuid.uuid4().hex
    metadata = dict(
        id=identifier,
        name=name,
        target=target,
        unit=unit,
        maximize=maximize,
        category=category,
        custom=True,
        ready=True,
        unique_sequences=len(unique),
        source_rows=len(records),
        duplicates_aggregated=len(records) - len(unique),
        sha256=hashlib.sha256(raw).hexdigest(),
        license="User-supplied data",
        publication_url="",
        source_url="",
        sequence_scope="User-supplied variable domains",
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{identifier}.json").write_text(
        json.dumps(dict(metadata=metadata, records=unique), allow_nan=False), encoding="utf-8"
    )
    return metadata
