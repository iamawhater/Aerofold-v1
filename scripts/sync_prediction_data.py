"""Package reviewed antibody assay tasks from a pinned, attributed FLAb snapshot.

The source catalog includes unavailable/restricted/unreviewed studies. Only
explicitly defined assays enter training; no generic fitness-column pooling is
performed. Re-run with --download to fetch missing public source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "e05e3f0135783ffb7de412df6a1467e7a0031911"
BASE = f"https://raw.githubusercontent.com/Graylab/FLAb/{COMMIT}/"
LICENSES = {"Attribution 4.0 International", "MIT License", "Clear BSD"}
AA = set("ACDEFGHIKLMNPQRSTVWY")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_source(path: str, download: bool) -> bytes:
    """Fetch an explicit repository path; never execute downloaded content."""
    destination = ROOT / "runtime/public-source/flab" / path
    if not destination.exists():
        if not download:
            raise ValueError(f"Missing source {path}; use --download")
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(BASE + path, headers={"User-Agent": "AeroFold-Ab/2.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read(60_000_001)
        if len(raw) > 60_000_000:
            raise ValueError("Source exceeds the importer size limit")
        destination.write_bytes(raw)
    return destination.read_bytes()


def task_specs() -> list[dict]:
    """Explicit units are taken from raw columns, not ambiguous catalog arrows."""
    specs = [
        dict(
            id="titeseq_4420_binding",
            name="4-4-20 · fluorescein binding",
            target="Fluorescein / 4-4-20",
            category="binding",
            file="adams2017measuring_4420-fluorescein_kd-titeseq.csv",
            column="Kd (Tite-Seq) [M]",
            transform="pkd_m",
            unit="pKd (M)",
            maximize=True,
            bounds=[5.0, 9.500313],
        ),
        dict(
            id="titeseq_4420_expression",
            name="4-4-20 · yeast expression",
            target="4-4-20 / yeast display",
            category="expression",
            file="adams2017measuring_4420-fluorescein_exp_er.csv",
            column="Expression [ER]",
            transform="identity",
            unit="Expression ratio (ER)",
            maximize=True,
        ),
        dict(
            id="cr9114_h1_binding",
            name="CR9114 · influenza H1 binding",
            target="Influenza H1 / CR9114",
            category="binding",
            file="phillips2021binding_cr9114_h1_kd.csv",
            column="h1_mean_neg_log_kd",
            transform="identity",
            unit="pKd (M)",
            maximize=True,
            lower_detection_limit=7.0,
        ),
        dict(
            id="cr9114_h3_binding",
            name="CR9114 · influenza H3 binding",
            target="Influenza H3 / CR9114",
            category="binding",
            file="phillips2021binding_cr9114_h3_kd.csv",
            column="fitness",
            transform="identity",
            unit="pKd (M)",
            maximize=True,
            lower_detection_limit=6.0,
        ),
        dict(
            id="d44_lysozyme_binding",
            name="D44 · lysozyme binding",
            target="Lysozyme / D44",
            category="binding",
            file="warszawski2019_d44_Kd.csv",
            column="KD (nM)",
            transform="pkd_nm",
            unit="pKd (M)",
            maximize=True,
        ),
        dict(
            id="trastuzumab_her2_binding",
            name="Trastuzumab variants · HER2",
            target="Human HER2 / trastuzumab scaffold",
            category="binding",
            file="shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv",
            column="KD (nM)",
            transform="pkd_nm",
            unit="pKd (M)",
            maximize=True,
        ),
    ]
    for antibody, target in (
        ("Afasevikumab", "IL17A"),
        ("Bimagrumab", "ACVR2B"),
        ("Eculizumab", "C5"),
        ("Osocimab", "FXI"),
        ("Spesolimab", "IL36R"),
        ("Tezepelumab", "TSLP"),
        ("Utomilumab", "TNFRSF9"),
    ):
        specs.append(
            dict(
                id=f"igdesign_{target.lower()}_binding",
                name=f"{antibody} variants · {target}",
                target=f"{target} / {antibody}",
                category="binding",
                file=f"shanehsazzadeh2024igdesign_{antibody}-{target}_kd.csv",
                column="KD (nM)",
                transform="pkd_nm",
                unit="pKd (M)",
                maximize=True,
            )
        )
    for key, method in (
        ("cd", "Circular dichroism"),
        ("dsf", "DSF (SYPRO)"),
        ("nanodsf", "nanoDSF"),
        ("dsc", "Diferencial scanning calorimetric (DSC)"),
    ):
        specs.append(
            dict(
                id=f"nbthermo_{key}",
                name=f"Nanobody thermostability · {key.upper()}",
                target=f"Nanobody VHH / {method}",
                category="thermostability",
                file="tresanco2023nbthermo_tm.csv",
                column="Tm measurement",
                transform="identity",
                unit="Melting temperature (°C)",
                maximize=True,
                filter_column="Tm Type",
                filter_value=method,
            )
        )
    return specs


def normalize(spec: dict, raw: bytes) -> tuple[list[dict], dict]:
    """Validate domains and aggregate identical sequence pairs before splitting."""
    records = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    rejected = Counter()
    for row in records:
        if spec.get("filter_column") and row.get(spec["filter_column"]) != spec["filter_value"]:
            rejected["different_assay_protocol"] += 1
            continue
        heavy = row.get("heavy", "").strip().upper()
        light = row.get("light", "").strip().upper()
        if not 70 <= len(heavy) <= 180 or (light and not 70 <= len(light) <= 180) or set(heavy + light) - AA:
            rejected["noncanonical_or_incomplete_domains"] += 1
            continue
        try:
            value = float(row[spec["column"]])
            if not math.isfinite(value):
                raise ValueError("Nonfinite score")
            if spec["transform"] in {"pkd_m", "pkd_nm"}:
                if value <= 0:
                    raise ValueError("Nonpositive affinity")
                value = -math.log10(value) + (9 if spec["transform"] == "pkd_nm" else 0)
        except (ValueError, KeyError):
            rejected["invalid_or_missing_measurement"] += 1
            continue
        groups[(heavy, light)].append(value)
    normalized = []
    for (heavy, light), values in sorted(groups.items()):
        sequence_hash = digest((heavy + "|" + light).encode())
        normalized.append(
            dict(
                variant_id="SEQ-" + sequence_hash[:16],
                heavy_sequence=heavy,
                light_sequence=light,
                score=statistics.median(values),
                replicate_count=len(values),
                sequence_sha256=sequence_hash,
            )
        )
    return normalized, {
        "source_rows": len(records),
        "unique_sequences": len(normalized),
        "duplicates_aggregated": sum(len(v) - 1 for v in groups.values()),
        "excluded_rows": dict(rejected),
    }


def main(download: bool = False) -> None:
    output = ROOT / "data/public"
    (output / "tasks").mkdir(parents=True, exist_ok=True)
    metadata_raw = read_source("data/flab_metadata.csv", download)
    metadata = list(csv.DictReader(io.StringIO(metadata_raw.decode("utf-8-sig"))))
    by_name = {r["filename"]: r for r in metadata}
    catalog, tasks = [], []
    admitted = {s["file"] for s in task_specs()}
    for source in metadata:
        entry = dict(source)
        entry["source_url"] = BASE + "data/" + source["category"] + "/" + source["filename"]
        entry["status"] = "reviewed" if source["filename"] in admitted else "not_admitted"
        entry["reason"] = (
            "Explicit assay adapter"
            if entry["status"] == "reviewed"
            else "Assay semantics, size, sequence completeness, or task suitability not reviewed"
        )
        if source["license"] not in LICENSES:
            entry["status"] = "restricted_or_unspecified_license"
            entry["reason"] = "Not included in the commercial-use training catalog"
        cached = ROOT / "runtime/public-source/flab/data" / source["category"] / source["filename"]
        if cached.exists():
            entry["source_sha256"] = digest(cached.read_bytes())
        catalog.append(entry)
    for spec in task_specs():
        source = by_name[spec["file"]]
        if source["license"] not in LICENSES:
            raise ValueError("Task is missing a reviewed source license")
        source_path = "data/" + spec["category"] + "/" + spec["file"]
        raw = read_source(source_path, download)
        records, qc = normalize(spec, raw)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "variant_id",
                "heavy_sequence",
                "light_sequence",
                "score",
                "replicate_count",
                "sequence_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
        encoded = buffer.getvalue().encode()
        path = output / "tasks" / (spec["id"] + ".csv")
        path.write_bytes(encoded)
        tasks.append(
            {
                **spec,
                **qc,
                "file": "tasks/" + path.name,
                "csv_sha256": digest(encoded),
                "source_sha256": digest(raw),
                "source_url": BASE + source_path,
                "source_commit": COMMIT,
                "publication_url": source["doi"],
                "citation": source["publication_title"],
                "license": source["license"],
                "license_url": source["license_link"],
                "ready": len(records) >= 60,
                "processing": "Complete canonical variable domains; exact sequence pairs collapsed to median transformed assay values; separate targets and measurement protocols; no label-based subsampling.",
                "sequence_scope": "Full published variable domains supplied by FLAb; framework sequence is not generated.",
            }
        )
        print(spec["id"], len(records), "unique records")
    document = {
        "version": 1,
        "source": "FLAb / Gray Laboratory",
        "source_commit": COMMIT,
        "metadata_sha256": digest(metadata_raw),
        "tasks": tasks,
        "sources": catalog,
        "additional_sources": [
            {
                "name": "Ginkgo GDPa1",
                "url": "https://huggingface.co/datasets/ginkgo-datapoints/GDPa1",
                "status": "not_imported_access_requires_account_and_terms",
            }
        ],
        "limitations": [
            "Catalogued data are not all interchangeable training observations.",
            "Foundation-model pretraining overlap cannot be ruled out.",
            "Assay replicate medians do not model censored measurements or laboratory batch effects.",
        ],
    }
    (output / "catalog.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "Packaged",
        len(tasks),
        "tasks;",
        sum(t["unique_sequences"] for t in tasks),
        "task-specific sequence/assay records",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    main(parser.parse_args().download)
