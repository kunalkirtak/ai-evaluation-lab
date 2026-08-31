"""Tests for src/metrics.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.config import Config
from src.embeddings import EmbeddingModel
from src.metrics import (
    answer_accuracy,
    answer_relevance,
    compute_generation_metrics,
    compute_retrieval_metrics,
    faithfulness,
    hallucination_rate,
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    semantic_similarity,
)


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingModel()


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b"]) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b"], ["c", "d"]) == 0.0

    def test_partial(self):
        assert precision_at_k(["a", "b", "c"], ["a"]) == pytest.approx(1 / 3)

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"]) == 0.0


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"]) == 1.0

    def test_none_found(self):
        assert recall_at_k(["c", "d"], ["a", "b"]) == 0.0

    def test_partial(self):
        assert recall_at_k(["a"], ["a", "b"]) == pytest.approx(0.5)

    def test_empty_relevant(self):
        assert recall_at_k(["a"], []) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        assert mean_reciprocal_rank(["x", "a", "c"], ["a"]) == pytest.approx(0.5)

    def test_not_found(self):
        assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0


class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k(["a", "x"], ["a"]) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["x", "y"], ["a"]) == 0.0


class TestRetrievalMetricsBundle:
    def test_compute_retrieval_metrics(self):
        metrics = compute_retrieval_metrics(["a", "b"], ["a"])
        assert metrics.precision_at_k == pytest.approx(0.5)
        assert metrics.recall_at_k == 1.0
        assert metrics.mrr == 1.0
        assert metrics.hit_rate_at_k == 1.0


class TestSemanticSimilarity:
    def test_identical_text_is_similar(self, embedder):
        score = semantic_similarity("the sky is blue", "the sky is blue", embedder)
        assert score > 0.9

    def test_unrelated_text_scores_lower_than_identical(self, embedder):
        same = semantic_similarity("the sky is blue", "the sky is blue", embedder)
        different = semantic_similarity("the sky is blue", "bananas are yellow fruit", embedder)
        assert same > different

    def test_empty_text_returns_zero(self, embedder):
        assert semantic_similarity("", "something", embedder) == 0.0

    def test_score_in_bounds(self, embedder):
        score = semantic_similarity("machine learning models", "deep neural networks", embedder)
        assert 0.0 <= score <= 1.0


class TestAnswerAccuracyRelevance:
    def test_answer_accuracy_high_for_close_match(self, embedder):
        score = answer_accuracy(
            "Machine learning is learning patterns from data.",
            "Machine learning is a field of AI where systems learn patterns from data.",
            embedder,
        )
        assert score > 0.5

    def test_answer_relevance_bounds(self, embedder):
        score = answer_relevance("What is RAG?", "RAG combines retrieval with generation.", embedder)
        assert 0.0 <= score <= 1.0


class TestFaithfulness:
    def test_supported_answer_has_high_faithfulness(self, embedder):
        context = ["The sky is blue because of Rayleigh scattering of sunlight."]
        answer = "The sky is blue due to Rayleigh scattering."
        score, unsupported, total = faithfulness(answer, context, embedder, support_threshold=0.4)
        assert total == 1
        assert score >= 0.0

    def test_no_context_gives_zero_faithfulness(self, embedder):
        score, unsupported, total = faithfulness("Some answer sentence here.", [], embedder)
        assert score == 0.0
        assert unsupported == total

    def test_empty_answer_returns_zero_totals(self, embedder):
        score, unsupported, total = faithfulness("", ["some context"], embedder)
        assert total == 0
        assert unsupported == 0


class TestHallucinationRate:
    def test_complement_of_faithfulness(self):
        assert hallucination_rate(0.8) == pytest.approx(0.2)
        assert hallucination_rate(1.0) == 0.0
        assert hallucination_rate(0.0) == 1.0


class TestGenerationMetricsBundle:
    def test_compute_generation_metrics_shapes(self, embedder):
        config = Config()
        metrics = compute_generation_metrics(
            question="What is RAG?",
            generated_answer="RAG retrieves context and generates an answer from it.",
            ground_truth="RAG combines retrieval with generation to answer questions.",
            context_chunks=["RAG combines retrieval with generation."],
            embedding_model=embedder,
            config=config,
        )
        assert 0.0 <= metrics.answer_accuracy <= 1.0
        assert 0.0 <= metrics.answer_relevance <= 1.0
        assert 0.0 <= metrics.faithfulness_score <= 1.0
        assert metrics.hallucination_rate == pytest.approx(1.0 - metrics.faithfulness_score)
