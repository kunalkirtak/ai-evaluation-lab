"""Embedding model wrapper.

Uses sentence-transformers (all-MiniLM-L6-v2 by default) when available.
If the library or model weights cannot be loaded (e.g. no internet access
in a restricted environment), the project falls back to a deterministic
hashing-based bag-of-words embedding so that the pipeline still runs
end-to-end without crashing. This fallback is clearly logged and is not
intended to match sentence-transformers quality -- it exists purely so the
evaluation harness remains runnable in constrained environments.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np

from src.logging_config import setup_logging

logger = setup_logging()

_FALLBACK_DIM = 384


class EmbeddingModel:
    """Lazy-loading embedding model with a deterministic offline fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._use_fallback = False

    def _load(self) -> None:
        if self._model is not None or self._use_fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("Loading sentence-transformers model '%s'...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded.")
        except Exception as exc:  # noqa: BLE001 - broad on purpose, this is a fallback path
            logger.warning(
                "Could not load sentence-transformers model (%s). "
                "Falling back to deterministic hashing embeddings.",
                exc,
            )
            self._use_fallback = True

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into an (N, D) array of L2-normalized vectors."""
        if not texts:
            return np.zeros((0, _FALLBACK_DIM), dtype=np.float32)

        self._load()

        if self._model is not None:
            embeddings = self._model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            embeddings = np.vstack([self._hashing_embed(t) for t in texts])

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    @staticmethod
    def _hashing_embed(text: str, dim: int = _FALLBACK_DIM) -> np.ndarray:
        """Deterministic bag-of-words hashing embedding used only as a fallback."""
        vector = np.zeros(dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % dim
            sign = 1.0 if int(digest, 16) % 2 == 0 else -1.0
            vector[index] += sign
        return vector


def cosine_similarity_matrix(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between each query vector and each doc vector.

    Assumes both inputs are already L2-normalized (as returned by
    EmbeddingModel.encode), so this reduces to a dot product.
    """
    if query_vecs.ndim == 1:
        query_vecs = query_vecs.reshape(1, -1)
    return query_vecs @ doc_vecs.T


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (need not be normalized)."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
