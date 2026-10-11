"""Local descriptors and optional real ESM-2 embeddings with masked pseudo-perplexity."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from backend.models import SequenceRecord

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AVILMFWY")


def developability(record: SequenceRecord) -> dict[str, float]:
    """Sequence-only heuristic liabilities, without asserting 3D stability or solubility."""
    chains = [s for s in (record.heavy_sequence, record.light_sequence) if s]
    charges, patches, aprs = [], [], []
    for sequence in chains:
        charge = np.array(
            [1.0 if a in "KR" else 0.1 if a == "H" else -1.0 if a in "DE" else 0.0 for a in sequence]
        )
        hydro = np.array([float(a in HYDROPHOBIC) for a in sequence])
        charges.append(float(charge.sum()))
        patches.extend(float(np.mean(hydro[i : i + 7])) for i in range(len(sequence) - 6))
        aprs.extend(
            float(np.mean(hydro[i : i + 7]) * (1 - min(1, np.mean(np.abs(charge[i : i + 7])))))
            for i in range(len(sequence) - 6)
        )
    seq = "".join(chains)
    return {
        "net_charge": round(sum(charges), 3),
        "max_chain_abs_charge": round(max(abs(c) for c in charges), 3),
        "hydrophobic_patch": round(max(patches), 4),
        "apr": round(max(aprs), 4),
        "charge_patch": float(
            max(
                max(
                    abs(sum((1 if a in "KR" else -1 if a in "DE" else 0) for a in s[i : i + 7]))
                    for i in range(len(s) - 6)
                )
                for s in chains
            )
        ),
        "hydrophobic_fraction": sum(a in HYDROPHOBIC for a in seq) / len(seq),
    }


def descriptor_embeddings(records: list[SequenceRecord]) -> np.ndarray:
    """Seeded position-sensitive one-hot projection; explicitly NOT a pretrained pLM."""
    embeddings = positional_embeddings([(r.heavy_sequence, r.light_sequence) for r in records])
    rows = []
    for record, embedding in zip(records, embeddings):
        props = developability(record)
        rows.append(np.concatenate([embedding, np.array(list(props.values()), dtype=np.float32) / 10]))
    return np.stack(rows).astype(np.float64)


def positional_embeddings(segments: list[tuple[str, str]]) -> np.ndarray:
    """Encode two separately positioned segments without invented intervening sequence.

    Discovery supplies heavy/light domains; the benchmark supplies CDR1H/CDR3H
    fragments. The segment interpretation must be recorded by the caller.
    """
    rng = np.random.default_rng(2026)
    projection = rng.normal(0, 1 / np.sqrt(96), size=(2, 180, 20, 96)).astype(np.float32)
    rows = []
    for pair in segments:
        embedding = np.zeros(96, dtype=np.float32)
        for chain, sequence in enumerate(pair):
            if len(sequence) > 180 or set(sequence) - set(ALPHABET):
                raise ValueError("Invalid sequence segment for positional encoding")
            for pos, residue in enumerate(sequence):
                embedding += projection[chain, pos, ALPHABET.index(residue)]
        rows.append(embedding / 10)
    return np.stack(rows).astype(np.float64)


def fragment_mutation_embeddings(sequences: list[str], parent: str) -> np.ndarray:
    """Exact position/residue mutation contrasts for a fixed-length DMS library.

    Each site has its own 20-amino-acid block. A substitution contributes +1
    for the mutant and -1 for the parent residue, preserving the additive
    mutational signal that a random projection can obscure. No framework or
    unseen assay value is inferred from these fragments.
    """
    if not parent or set(parent) - set(ALPHABET):
        raise ValueError("Parent must contain canonical amino acids")
    if any(len(s) != len(parent) or set(s) - set(ALPHABET) for s in sequences):
        raise ValueError("DMS fragments must match the canonical parent length")
    encoded = np.zeros((len(sequences), len(parent), len(ALPHABET)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        for position, (reference, residue) in enumerate(zip(parent, sequence)):
            if residue != reference:
                encoded[row, position, ALPHABET.index(residue)] = 1.0
                encoded[row, position, ALPHABET.index(reference)] = -1.0
    midpoint = len(parent) // 2
    counts = np.array(
        [
            [
                sum(a != b for a, b in zip(sequence[:midpoint], parent[:midpoint])),
                sum(a != b for a, b in zip(sequence[midpoint:], parent[midpoint:])),
            ]
            for sequence in sequences
        ],
        dtype=np.float64,
    )
    return np.concatenate((encoded.reshape(len(sequences), -1), 2.0 * counts), axis=1)


class ESMEmbedder:
    """Mean-pool real ESM-2 hidden states and evaluate each masked residue once.

    Cache entries are keyed by model + exact sequence. No fabricated likelihoods
    or fallback descriptors are returned when model loading fails.
    """

    model_name = "facebook/esm2_t6_8M_UR50D"
    revision = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"

    def __init__(self, cache_dir: Path):
        try:
            from transformers import AutoTokenizer, EsmForMaskedLM
        except ImportError as exc:
            raise ValueError("ESM-2 requires: pip install -r requirements-esm.txt") from exc
        local_only = os.getenv("AEROFOLD_ESM_LOCAL_ONLY", "0") == "1"
        model_cache = Path(os.getenv("HF_HOME", str(cache_dir.parent / "huggingface"))) / "hub"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision, cache_dir=model_cache, local_files_only=local_only
        )
        self.model = EsmForMaskedLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            cache_dir=model_cache,
            local_files_only=local_only,
            use_safetensors=True,
        ).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def chain(self, sequence: str) -> tuple[np.ndarray, float]:
        """Return residue mean embedding and mean masked negative log likelihood."""
        key = hashlib.sha256((self.model_name + ":" + self.revision + ":v1:" + sequence).encode()).hexdigest()
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                return cached["embedding"], float(cached["nll"])
        tokens = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**tokens, output_hidden_states=True)
            embedding = output.hidden_states[-1][0, 1 : len(sequence) + 1].mean(0).cpu().numpy()
            losses = []
            for start in range(1, len(sequence) + 1, 16):
                positions = torch.arange(start, min(start + 16, len(sequence) + 1), device=self.device)
                count = len(positions)
                ids = tokens["input_ids"].repeat(count, 1)
                targets = ids[torch.arange(count, device=self.device), positions].clone()
                ids[torch.arange(count, device=self.device), positions] = self.tokenizer.mask_token_id
                logits = self.model(
                    input_ids=ids, attention_mask=tokens["attention_mask"].repeat(count, 1)
                ).logits
                loss = (
                    -torch.log_softmax(logits[torch.arange(count, device=self.device), positions], dim=-1)
                    .gather(1, targets[:, None])
                    .squeeze(1)
                )
                losses.extend(loss.cpu().tolist())
        nll = float(np.mean(losses))
        np.savez_compressed(path, embedding=embedding, nll=nll)
        return embedding, nll

    def encode(self, records: list[SequenceRecord]) -> tuple[np.ndarray, list[float]]:
        """Concatenate chain embeddings; length-weight mean masked NLL across chains."""
        embeddings, perplexities = [], []
        for record in records:
            vectors, losses, lengths = [], [], []
            for seq in (record.heavy_sequence, record.light_sequence):
                if seq:
                    vector, nll = self.chain(seq)
                    vectors.append(vector)
                    losses.append(nll)
                    lengths.append(len(seq))
            embeddings.append(np.concatenate(vectors))
            perplexities.append(float(np.exp(np.average(losses, weights=lengths))))
        return np.stack(embeddings).astype(np.float64), perplexities


class AbLang2Embedder:
    """Encode complete antibody variable domains with the paired AbLang-2 model.

    The vendor's sequence-coding API pools antibody-specific residue states.
    Cache keys include both chains, so partner-chain context is preserved. This
    encoder deliberately rejects CDR-only fragments rather than inventing a
    framework. It does not claim a masked naturalness score.
    """

    model_name = "ablang2-paired"
    package_version = "0.1.1"

    def __init__(self, cache_dir: Path):
        try:
            import ablang2
        except ImportError as exc:
            raise ValueError("AbLang-2 requires: pip install -r requirements-ablang2.txt") from exc
        self.model = ablang2.pretrained(
            model_to_use=self.model_name,
            random_init=False,
            ncpu=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def encode(self, records: list[SequenceRecord]) -> np.ndarray:
        """Return one cached paired embedding per full-domain sequence record."""
        if any(
            (sequence and not 70 <= len(sequence) <= 180)
            for record in records
            for sequence in (record.heavy_sequence, record.light_sequence)
        ):
            raise ValueError("AbLang-2 requires full 70–180 residue variable domains")
        paths = [
            self.cache_dir
            / f"{hashlib.sha256((self.model_name + ':' + self.package_version + ':' + r.heavy_sequence + '|' + r.light_sequence).encode()).hexdigest()}.npz"
            for r in records
        ]
        vectors: list[np.ndarray | None] = [None] * len(records)
        pending = []
        for index, path in enumerate(paths):
            if path.exists():
                with np.load(path, allow_pickle=False) as cached:
                    vectors[index] = cached["embedding"]
            else:
                pending.append(index)
        for offset in range(0, len(pending), 16):
            indices = pending[offset : offset + 16]
            pairs = [[records[i].heavy_sequence, records[i].light_sequence] for i in indices]
            with torch.inference_mode():
                batch = np.asarray(self.model(pairs, mode="seqcoding", batch_size=16), dtype=np.float64)
            if batch.ndim == 1 and len(indices) == 1:
                batch = batch[None, :]
            if batch.ndim != 2 or len(batch) != len(indices) or not np.isfinite(batch).all():
                raise ValueError("AbLang-2 returned invalid sequence embeddings")
            for index, vector in zip(indices, batch):
                np.savez_compressed(paths[index], embedding=vector)
                vectors[index] = vector
        return np.stack(vectors).astype(np.float64)
