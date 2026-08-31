"""Tests for src/evaluator.py (integration-style, using the real data)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.chunker import chunk_documents
from src.config import Config
from src.data_loader import load_documents, load_evaluation_dataset
from src.embeddings import EmbeddingModel
from src.evaluator import EvaluationEngine, classify_failure
from src.generator import AnswerGenerator
from src.retriever import Retriever
from src.config import DOCUMENTS_DIR, EVAL_DATASET_PATH


@pytest.fixture(scope="module")
def config():
    return Config(top_k=3, experiment_name="test")


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingModel()


@pytest.fixture(scope="module")
def engine(config, embedder):
    documents = load_documents(DOCUMENTS_DIR)
    chunks = chunk_documents(
        documents,
        chunk_size_words=config.chunk_size_words,
        overlap_words=config.chunk_overlap_words,
    )
    retriever = Retriever(chunks, embedder)
    generator = AnswerGenerator(config, embedder)
    return EvaluationEngine(config, retriever, generator, embedder)


@pytest.fixture(scope="module")
def questions():
    return load_evaluation_dataset(EVAL_DATASET_PATH)


class TestEvaluationEngine:
    def test_evaluate_question_returns_populated_result(self, engine, questions):
        result = engine.evaluate_question(questions[0])
        assert result.question_id == questions[0].id
        assert result.generated_answer != ""
        assert len(result.retrieved) > 0
        assert 0.0 <= result.retrieval_metrics.precision_at_k <= 1.0
        assert 0.0 <= result.generation_metrics.faithfulness_score <= 1.0
        assert result.tokens.total_tokens > 0
        assert result.latency.total_seconds >= 0.0

    def test_evaluate_question_assigns_failure_category(self, engine, questions):
        result = engine.evaluate_question(questions[0])
        assert result.failure_category in {
            "retrieval_failure",
            "unsupported_answer",
            "low_relevance",
            "generation_failure",
            "correct_answer",
        }

    def test_evaluate_all_returns_result_per_question(self, engine, questions):
        subset = questions[:4]
        results = engine.evaluate_all(subset)
        assert len(results) == len(subset)
        returned_ids = {r.question_id for r in results}
        expected_ids = {q.id for q in subset}
        assert returned_ids == expected_ids

    def test_easy_question_retrieves_its_own_document(self, engine, questions):
        # q001 asks "What is machine learning?" and should retrieve doc_001.
        q001 = next(q for q in questions if q.id == "q001")
        result = engine.evaluate_question(q001)
        retrieved_doc_ids = {r.doc_id for r in result.retrieved}
        assert "doc_001" in retrieved_doc_ids


class TestClassifyFailure:
    def test_retrieval_failure_when_no_hit(self, config, engine, questions):
        result = engine.evaluate_question(questions[0])
        result.retrieval_metrics.hit_rate_at_k = 0.0
        assert classify_failure(result, config) == "retrieval_failure"
