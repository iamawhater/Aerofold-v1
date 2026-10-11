"""Optional integration smoke check against real ESM-2 weights, not a mocked model."""

from pathlib import Path

import numpy as np
import torch

from backend.features import ESMEmbedder
from backend.models import SequenceRecord
from backend.sequences import DEMO_HEAVY

if __name__ == "__main__":
    torch.set_num_threads(2)
    embedder = ESMEmbedder(Path("runtime/embeddings"))
    embeddings, perplexities = embedder.encode(
        [SequenceRecord(variant_id="parent", heavy_sequence=DEMO_HEAVY)]
    )
    assert embeddings.shape == (1, 320)
    assert np.isfinite(embeddings).all()
    assert np.isfinite(perplexities).all() and perplexities[0] > 1
    cached, cached_ppl = embedder.encode([SequenceRecord(variant_id="parent", heavy_sequence=DEMO_HEAVY)])
    np.testing.assert_array_equal(embeddings, cached)
    assert perplexities == cached_ppl
    print(
        f"Real ESM-2 verified: {embeddings.shape}, masked pseudo-perplexity={perplexities[0]:.4f}; cache matches."
    )
