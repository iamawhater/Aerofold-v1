"""Package a reproducible, score-independent subset of public Tite-Seq data.

Run explicitly to refresh artifacts; the application never downloads data at runtime.
Only CSV measurement data are read. No upstream analysis code is executed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path

COMMIT = "54130c8fc6081bbfa8301c41e1afcfa6a5d79590"
BASE = f"https://raw.githubusercontent.com/jbkinney/16_titeseq/{COMMIT}"
PARENT = "TFSDYWMNWVGSYYGMDYWG"
ROOT = Path(__file__).resolve().parents[1]


def package() -> None:
    """Median replicate estimates, group synonymous variants, then hash-sample 512."""
    cache = ROOT / "runtime" / "benchmark-source"
    target = ROOT / "data" / "benchmarks"
    cache.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    observations: dict[str, list[tuple[float, float]]] = defaultdict(list)
    sources = []
    for replicate in (1, 2, 3):
        name = f"replicate_{replicate}.csv"
        path = cache / f"{COMMIT}-{name}"
        url = f"{BASE}/data/{name}"
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        data = path.read_bytes()
        sources.append({"url": url, "sha256": hashlib.sha256(data).hexdigest()})
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in csv.DictReader(io.StringIO(data.decode())):
            sequence = row["CDR1H_AA"] + row["CDR3H_AA"]
            if len(sequence) != 20 or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                continue
            try:
                kd, expression = float(row["fit_KD"]), float(row["expression"])
            except ValueError:
                continue
            if not math.isfinite(kd) or not math.isfinite(expression) or kd <= 0 or expression < 0:
                continue
            grouped[sequence].append((min(9.5, max(5.0, -math.log10(kd))), expression))
        wt_expression = statistics.median(value[1] for value in grouped[PARENT])
        for sequence, values in grouped.items():
            binding = statistics.median(v[0] for v in values)
            relative_expression = statistics.median(v[1] for v in values) / wt_expression
            observations[sequence].append((binding, relative_expression))
    candidates = [seq for seq, values in observations.items() if seq != PARENT and len(values) >= 2]
    candidates.sort(key=lambda seq: hashlib.sha256(("aerofold-titeseq-v1:" + seq).encode()).hexdigest())
    selected = sorted(candidates[:512])
    if len(selected) != 512:
        raise ValueError("Insufficient public variants after QC")
    output = io.StringIO(newline="")
    fields = [
        "variant_id",
        "parent_sequence",
        "variant_sequence",
        "variant_string",
        "true_binding",
        "stability_score",
        "relative_expression",
        "replicate_count",
        "stability_kind",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for i, seq in enumerate(selected):
        values = observations[seq]
        expression = statistics.median(v[1] for v in values)
        mutations = [
            f"{'CDR1H' if pos < 10 else 'CDR3H'}:{a}{pos % 10 + 1}{b}"
            for pos, (a, b) in enumerate(zip(PARENT, seq))
            if a != b
        ]
        writer.writerow(
            {
                "variant_id": f"TITE-{i + 1:04d}",
                "parent_sequence": PARENT,
                "variant_sequence": seq,
                "variant_string": ";".join(mutations),
                "true_binding": f"{statistics.median(v[0] for v in values):.8f}",
                "stability_score": f"{100 * expression / (1 + expression):.8f}",
                "relative_expression": f"{expression:.8f}",
                "replicate_count": len(values),
                "stability_kind": "expression_proxy_not_structural_stability",
            }
        )
    content = output.getvalue().encode()
    (target / "titeseq_4_4_20.csv").write_bytes(content)
    metadata = {
        "id": "titeseq_4_4_20",
        "name": "4-4-20 antibody · Tite-Seq",
        "description": "512 public antibody variants with measured fluorescein affinity and expression, aggregated across experimental replicates.",
        "citation": "Adams RM, Mora T, Walczak AM, Kinney JB (2016). Measuring the sequence-affinity landscape of antibodies with massively parallel titration curves. eLife 5:e23156.",
        "publication_url": "https://doi.org/10.7554/eLife.23156",
        "source_url": "https://github.com/jbkinney/16_titeseq",
        "source_commit": COMMIT,
        "sources": sources,
        "csv_sha256": hashlib.sha256(content).hexdigest(),
        "variant_count": len(selected),
        "qc_population_size": len(candidates),
        "sequence_scope": "Concatenated assayed 10-residue CDR1H and CDR3H fragments; not a full variable-domain sequence. Mutation coordinates are fragment-local.",
        "parent_sequence": PARENT,
        "binding_label": "Binding pKd (higher is better)",
        "stability_label": "Expression-based developability proxy (not structural stability)",
        "processing": "Canonical 20-aa fragment pairs; finite positive Kd and nonnegative expression; pKd clipped to [5,9.5] assay range; median synonymous measurements within each replicate; expression divided by replicate WT median; require >=2 replicates; median across replicates; deterministic SHA256-ranked 512-variant subset independent of outcome scores. stability_score=100*relative_expression/(1+relative_expression).",
        "redistribution_note": "Public factual measurement subset, attributed to Adams et al. No upstream code is redistributed. The associated eLife article is CC BY 4.0; the source repository has no separate license file. Retain attribution and provenance.",
        "warnings": [
            "Expression is a developability proxy, not a measured folding-stability endpoint.",
            "Binding fits are clipped at assay limits; top-5% cutoff includes ties.",
            "This subset and one random seed do not establish general superiority or experimental validity.",
        ],
    }
    (target / "titeseq_4_4_20.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Packaged {len(selected)} variants from {len(candidates)} QC-passing unique variants; SHA256 {metadata['csv_sha256']}"
    )


if __name__ == "__main__":
    package()
