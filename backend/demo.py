"""Reproducible synthetic benchmark; hidden outcomes never enter model fitting."""

from __future__ import annotations

import csv
import io
import random

from backend.models import DatasetInput, Observation, SequenceRecord
from backend.sequences import DEMO_HEAVY, DEMO_LIGHT, annotate


def make_demo() -> tuple[DatasetInput, list[Observation]]:
    """Generate 500 unique CDR variants, 48 measured seeds and an isolated oracle."""
    rng = random.Random(2026)
    parent = SequenceRecord(variant_id="parent", heavy_sequence=DEMO_HEAVY, light_sequence=DEMO_LIGHT)
    positions = []
    for chain, sequence in (("H", DEMO_HEAVY), ("L", DEMO_LIGHT)):
        for region in annotate(sequence, chain)["regions"]:
            positions.extend((chain, i - 1) for i in range(region["start"], region["end"] + 1))
    alphabet = "ADEFGHIKLMNPQRSTVWY"
    effects = {
        (chain, i, aa): rng.gauss(0, 0.28) + (0.16 if aa in "YWR" else -0.04)
        for chain, i in positions
        for aa in alphabet
    }
    variants, oracle, seen = [], [], {(DEMO_HEAVY, DEMO_LIGHT)}
    while len(variants) < 500:
        heavy, light = list(DEMO_HEAVY), list(DEMO_LIGHT)
        selected = rng.sample(positions, rng.choices(range(1, 8), weights=[1, 2, 3, 3, 2, 1, 1])[0])
        gain, burden = 0.0, 0.0
        for chain, pos in selected:
            seq = heavy if chain == "H" else light
            aa = rng.choice(alphabet.replace(seq[pos], ""))
            gain += effects[(chain, pos, aa)]
            burden += 1.5 + (1.3 if aa in "FWIL" else 0) + (1.0 if aa == "P" else 0)
            seq[pos] = aa
        pair = "".join(heavy), "".join(light)
        if pair in seen:
            continue
        seen.add(pair)
        variant_id = f"AB-{len(variants) + 1:04d}"
        variants.append(SequenceRecord(variant_id=variant_id, heavy_sequence=pair[0], light_sequence=pair[1]))
        oracle.append(
            Observation(
                variant_id=variant_id,
                binding=round(7.0 + gain + rng.gauss(0, 0.07), 4),
                stability=round(87.0 - burden - gain * 1.3 + rng.gauss(0, 0.8), 3),
            )
        )
    measured = set(rng.sample(range(500), 48))
    observations = [Observation(variant_id="parent", binding=7.0, stability=87.0)] + [
        o for i, o in enumerate(oracle) if i in measured
    ]
    return DatasetInput(
        name="CDR exploration · demo library", parent=parent, variants=variants, observations=observations
    ), oracle


def library_csv(dataset: DatasetInput, all_outcomes: list[Observation] | None = None) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["variant_id", "role", "heavy_sequence", "light_sequence", "binding", "stability"])
    outcomes = {o.variant_id: o for o in (all_outcomes if all_outcomes is not None else dataset.observations)}
    for record in [dataset.parent, *dataset.variants]:
        observation = outcomes.get(record.variant_id)
        writer.writerow(
            [
                record.variant_id,
                "parent" if record == dataset.parent else "variant",
                record.heavy_sequence,
                record.light_sequence,
                observation.binding if observation else "",
                observation.stability if observation else "",
            ]
        )
    return output.getvalue()
