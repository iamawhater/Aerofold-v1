"""Strict FASTA/CSV parsing and transparent reference-aligned CDR annotation.

CDR intervals are approximate, template-relative coordinates, NOT IMGT numbering.
Insertions/deletions are retained through a global alignment to the parent.
"""

from __future__ import annotations

import csv
import io
from functools import lru_cache

from backend.models import DatasetInput, Observation, SequenceRecord

REFERENCE_PARTS = {
    "H": (
        "EVQLVESGGGLVQPGGSLRLSCAAS",
        "GFNIKDTYIH",
        "WVRQAPGKGLEWVA",
        "RIYPTNGYTRYADSVKG",
        "RFTISADTSKNTAYLQMNSLRAEDTAVYYC",
        "SRWGGDGFYAMDY",
        "WGQGTLVTVSS",
    ),
    "L": (
        "DIQMTQSPSSLSASVGDRVTITC",
        "RASQDVNTAVA",
        "WYQQKPGKAPKLLIY",
        "SASFLYS",
        "GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC",
        "QQHYTTPPT",
        "FGQGTKVEIK",
    ),
}
DEMO_HEAVY = "".join(REFERENCE_PARTS["H"])
DEMO_LIGHT = "".join(REFERENCE_PARTS["L"])


@lru_cache(maxsize=8192)
def align(reference: str, sequence: str) -> tuple[str, str]:
    """Needleman–Wunsch global alignment with deterministic traceback."""
    if reference == sequence:
        return reference, sequence
    n, m = len(reference), len(sequence)
    scores = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        scores[i][0] = -3 * i
    for j in range(m + 1):
        scores[0][j] = -3 * j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores[i][j] = max(
                scores[i - 1][j - 1] + (2 if reference[i - 1] == sequence[j - 1] else -1),
                scores[i - 1][j] - 3,
                scores[i][j - 1] - 3,
            )
    a, b = [], []
    i, j = n, m
    while i or j:
        if (
            i
            and j
            and scores[i][j] == scores[i - 1][j - 1] + (2 if reference[i - 1] == sequence[j - 1] else -1)
        ):
            i, j = i - 1, j - 1
            a.append(reference[i])
            b.append(sequence[j])
        elif i and scores[i][j] == scores[i - 1][j] - 3:
            i -= 1
            a.append(reference[i])
            b.append("-")
        else:
            j -= 1
            a.append("-")
            b.append(sequence[j])
    return "".join(reversed(a)), "".join(reversed(b))


def annotate(sequence: str, chain: str) -> dict:
    """Transfer approximate CDRs from a documented reference; reject poor homology."""
    parts = REFERENCE_PARTS[chain]
    reference = "".join(parts)
    a, b = align(reference, sequence)
    identity = sum(x == y for x, y in zip(a, b)) / max(len(reference), len(sequence))
    if identity < 0.60:
        raise ValueError(
            f"{chain} parent has only {identity:.0%} reference identity; reliable CDR transfer unavailable. Use a homologous variable-domain library."
        )
    labels = []
    for k, part in enumerate(parts):
        labels.extend([f"CDR{(k + 1) // 2}" if k % 2 else "FR"] * len(part))
    mapped, ref_i, seq_i = [], 0, 0
    for x, y in zip(a, b):
        label = labels[min(ref_i, len(labels) - 1)]
        if y != "-":
            seq_i += 1
            mapped.append({"position": seq_i, "residue": y, "region": label})
        if x != "-":
            ref_i += 1
    regions = []
    for label in ("CDR1", "CDR2", "CDR3"):
        positions = [p["position"] for p in mapped if p["region"] == label]
        if not positions:
            raise ValueError(f"Cannot transfer {chain} {label}")
        regions.append(
            {
                "name": f"{chain}-{label}",
                "start": min(positions),
                "end": max(positions),
                "sequence": sequence[min(positions) - 1 : max(positions)],
            }
        )
    return {
        "chain": chain,
        "method": "approximate reference alignment (not IMGT/Kabat)",
        "identity": round(identity, 3),
        "regions": regions,
    }


def mutations(parent: str, sequence: str, chain: str, annotation: dict) -> list[dict]:
    """Return substitutions and indels in one-based parent coordinates."""
    a, b = align(parent, sequence)
    position, insertion = 0, 0
    result = []
    for x, y in zip(a, b):
        if x != "-":
            position += 1
            insertion = 0
        else:
            insertion += 1
        if x != y:
            region = next(
                (r["name"] for r in annotation["regions"] if r["start"] <= position <= r["end"]),
                f"{chain}-FR",
            )
            suffix = f"+{insertion}" if insertion else ""
            result.append(
                {
                    "chain": chain,
                    "position": position,
                    "insertion": insertion,
                    "from": x,
                    "to": y,
                    "region": region,
                    "label": f"{chain}:{x}{position}{suffix}{y}",
                }
            )
    return result


def parse_library(content: bytes, filename: str, name: str = "Uploaded library") -> DatasetInput:
    """Parse a library with a parent row/header. Paired FASTA IDs use /H and /L."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Upload a UTF-8 FASTA or CSV file") from exc
    observations, parent = [], None
    if filename.lower().endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or "variant_id" not in reader.fieldnames:
            raise ValueError("CSV requires variant_id and sequence (or heavy_sequence/light_sequence)")
        rows = []
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("CSV has inconsistent column counts")
            chain_type = row.get("chain_type", "H").strip().upper()
            if chain_type not in {"H", "L"}:
                raise ValueError("chain_type must be H or L")
            single = row.get("sequence", "").strip()
            record = SequenceRecord(
                variant_id=row["variant_id"].strip(),
                heavy_sequence=row.get("heavy_sequence", "") or (single if chain_type == "H" else ""),
                light_sequence=row.get("light_sequence", "") or (single if chain_type == "L" else ""),
            )
            if row.get("role", "").lower() == "parent" or record.variant_id.lower() == "parent":
                if parent:
                    raise ValueError("Exactly one parent is required")
                parent = record
            else:
                rows.append(record)
            binding, stability = row.get("binding", "").strip(), row.get("stability", "").strip()
            if bool(binding) != bool(stability):
                raise ValueError(
                    "Measured rows require both binding and stability; leave both blank for candidates"
                )
            if binding:
                observations.append(
                    Observation(
                        variant_id=record.variant_id, binding=float(binding), stability=float(stability)
                    )
                )
    elif filename.lower().endswith((".fasta", ".fa", ".faa")):
        entries, header, chunks = {}, None, []
        for line in [s.strip() for s in text.splitlines() if s.strip()]:
            if line.startswith(">"):
                if header:
                    entries[header] = "".join(chunks)
                header = line[1:].split()[0] if line[1:].strip() else ""
                if not header or header in entries:
                    raise ValueError("FASTA headers must be nonempty and unique")
                chunks = []
            elif header is None:
                raise ValueError("FASTA must start with >parent or >parent/H")
            else:
                chunks.append(line)
        if header:
            entries[header] = "".join(chunks)
        grouped = {}
        for key, seq in entries.items():
            base, chain = key.rsplit("/", 1) if "/" in key else (key, "H")
            if chain not in {"H", "L"}:
                raise ValueError("FASTA chain suffix must be /H or /L")
            field = "heavy_sequence" if chain == "H" else "light_sequence"
            if field in grouped.setdefault(base, {}):
                raise ValueError("Duplicate FASTA chain")
            grouped[base][field] = seq
        records = [SequenceRecord(variant_id=k, **v) for k, v in grouped.items()]
        parents = [r for r in records if r.variant_id.lower() == "parent"]
        if len(parents) == 1:
            parent = parents[0]
        rows = [r for r in records if r.variant_id.lower() != "parent"]
    else:
        raise ValueError("Supported file extensions: .csv, .fasta, .fa, .faa")
    if parent is None:
        raise ValueError("Provide one parent: CSV role=parent / variant_id=parent, or FASTA >parent")
    return DatasetInput(name=name, parent=parent, variants=rows, observations=observations)
