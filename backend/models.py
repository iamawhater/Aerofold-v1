"""Validated domain objects. Objective scores always use higher-is-better units."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SequenceRecord(StrictModel):
    variant_id: str = Field(min_length=1, max_length=80)
    heavy_sequence: str = ""
    light_sequence: str = ""

    @field_validator("variant_id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(
                "IDs must start with a letter/number and contain only letters, numbers, _, . or -"
            )
        return value

    @field_validator("heavy_sequence", "light_sequence")
    @classmethod
    def valid_sequence(cls, value: str) -> str:
        value = "".join(value.split()).upper()
        if value and (not 70 <= len(value) <= 180 or set(value) - AA):
            raise ValueError(
                "Each chain must be a 70–180 residue variable domain using the 20 canonical amino acids"
            )
        return value

    @model_validator(mode="after")
    def has_chain(self) -> SequenceRecord:
        if not self.heavy_sequence and not self.light_sequence:
            raise ValueError("At least one heavy or light chain is required")
        return self


class Observation(StrictModel):
    variant_id: str
    binding: float = Field(ge=-1000, le=1000)
    stability: float = Field(ge=0, le=100)


class DatasetInput(StrictModel):
    name: str = Field(default="Antibody library", min_length=1, max_length=100)
    parent: SequenceRecord
    variants: list[SequenceRecord] = Field(min_length=24, max_length=2000)
    observations: list[Observation] = Field(default_factory=list, max_length=2001)

    @model_validator(mode="after")
    def consistent_library(self) -> DatasetInput:
        records = [self.parent, *self.variants]
        ids = [v.variant_id for v in records]
        if len(set(ids)) != len(ids):
            raise ValueError("Parent and variant IDs must be unique")
        pairs = [(v.heavy_sequence, v.light_sequence) for v in records]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate sequence pairs (including the parent) are not allowed")
        for v in self.variants:
            if (bool(v.heavy_sequence), bool(v.light_sequence)) != (
                bool(self.parent.heavy_sequence),
                bool(self.parent.light_sequence),
            ):
                raise ValueError("Every variant must have the same chain types as the parent")
        observed = [o.variant_id for o in self.observations]
        if len(set(observed)) != len(observed) or set(observed) - set(ids):
            raise ValueError("Observation IDs must be unique and present in the library")
        return self


class SelectionConfig(StrictModel):
    embedding: Literal["descriptor", "esm2", "ablang2"] = "descriptor"
    exploration: float = Field(default=0.35, ge=0, le=1)
    diversity: float = Field(default=0.55, ge=0, le=1)
    min_stability: float = Field(default=60, ge=0, le=100)
    confidence_beta: float = Field(default=1.0, ge=0, le=3)
    max_abs_charge: float = Field(default=16, ge=0, le=100)
    max_hydrophobic_patch: float = Field(default=0.9, ge=0, le=1)
    max_apr: float = Field(default=0.9, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    batch_size: Literal[24] = 24


class ObservationBatch(StrictModel):
    observations: list[Observation] = Field(min_length=1, max_length=2001)
