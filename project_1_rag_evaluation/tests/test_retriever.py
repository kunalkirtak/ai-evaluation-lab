"""Tests for src/retriever.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.embeddings import EmbeddingModel
from src.models import Chunk
from src.retriever import Retriever


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingModel()


@pytest.fixture(scope="module")
def sample_chunks():
    return [
        Chunk(chunk_id="c1", doc_id="doc_a", text="Cats are small furry domesticated animals."),
        Chunk(chunk_id="c2", doc_id="doc_b", text="Python is a popular programming language."),
        Chunk(chunk_id="c3", doc_id="doc_c", text="The stock market fluctuates based on many factors."),
    ]


class TestRetriever:
    def test_raises_on_empty_chunks(self, embedder):
        with pytest.raises(ValueError):
            Retriever([], embedder)

    def test_retrieve_returns_top_k(self, sample_chunks, embedder):
        retriever = Retriever(sample_chunks, embedder)
        results = retriever.retrieve("Tell me about cats", top_k=2)
        assert len(results) == 2

    def test_retrieve_ranks_are_sequential(self, sample_chunks, embedder):
        retriever = Retriever(sample_chunks, embedder)
        results = retriever.retrieve("programming languages", top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    def test_scores_are_descending(self, sample_chunks, embedder):
        retriever = Retriever(sample_chunks, embedder)
        results = retriever.retrieve("furry animals like cats", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_clamped_to_available_chunks(self, sample_chunks, embedder):
        retriever = Retriever(sample_chunks, embedder)
        results = retriever.retrieve("anything", top_k=100)
        assert len(results) == len(sample_chunks)

    def test_empty_query_raises(self, sample_chunks, embedder):
        retriever = Retriever(sample_chunks, embedder)
        with pytest.raises(ValueError):
            retriever.retrieve("   ", top_k=1)

    def test_relevant_chunk_is_top_ranked(self, sample_chunks, embedder):
        retriever = Retriever(sample_chunks, embedder)
        results = retriever.retrieve("What programming language is popular?", top_k=1)
        assert results[0].doc_id == "doc_b"
