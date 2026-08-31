"""Lightweight semantic retriever over an in-memory chunk index."""

from __future__ import annotations

from typing import List

import numpy as np

from src.embeddings import EmbeddingModel, cosine_similarity_matrix
from src.logging_config import setup_logging
from src.models import Chunk, RetrievalResult

logger = setup_logging()


class Retriever:
    """Embeds a fixed set of chunks and answers top-k similarity queries.

    Example:
        retriever = Retriever(chunks, embedding_model)
        results = retriever.retrieve("What is RAG?", top_k=3)
    """

    def __init__(self, chunks: List[Chunk], embedding_model: EmbeddingModel) -> None:
        if not chunks:
            raise ValueError("Retriever requires at least one chunk")
        self.chunks = chunks
        self.embedding_model = embedding_model
        self._chunk_vectors: np.ndarray = self.embedding_model.encode(
            [c.text for c in chunks]
        )
        logger.info("Retriever indexed %d chunks (dim=%d)", len(chunks), self._chunk_vectors.shape[1])

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Return the top_k most similar chunks to the query, ranked descending."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vec = self.embedding_model.encode([query])
        scores = cosine_similarity_matrix(query_vec, self._chunk_vectors)[0]

        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(-scores)[:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
        return results
