import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluator import SemanticEvaluator, evaluate_answer, _fallback_similarity
from src import config


def test_fallback_similarity_identical_text():
    score = _fallback_similarity("hello world", "hello world")
    assert score == 1.0


def test_fallback_similarity_disjoint_text():
    score = _fallback_similarity("apple banana", "car truck")
    assert score == 0.0


def test_fallback_similarity_empty_text():
    assert _fallback_similarity("", "hello") == 0.0


def test_semantic_evaluator_returns_score_in_range():
    evaluator = SemanticEvaluator()
    score = evaluator.similarity("What is a transformer?", "A transformer is a neural network architecture.")
    assert 0.0 <= score <= 1.0


def test_semantic_evaluator_identical_text_high_similarity():
    evaluator = SemanticEvaluator()
    text = "Retrieval-augmented generation combines retrieval and generation."
    score = evaluator.similarity(text, text)
    assert score > 0.9


def test_evaluate_answer_marks_correct_above_threshold():
    evaluator = SemanticEvaluator()
    reference = "A transformer is a neural network architecture based on self-attention."
    result = evaluate_answer(
        question="What is a transformer?",
        generated_answer=reference,
        reference_answer=reference,
        evaluator=evaluator,
    )
    assert result.is_correct is True
    assert result.quality_score >= config.QUALITY_THRESHOLD


def test_evaluate_answer_marks_incorrect_when_dissimilar():
    evaluator = SemanticEvaluator()
    result = evaluate_answer(
        question="What is a transformer?",
        generated_answer="Bananas are a good source of potassium and fiber.",
        reference_answer="A transformer is a neural network architecture based on self-attention.",
        evaluator=evaluator,
    )
    assert result.quality_score < 1.0


def test_evaluate_answer_no_judge_by_default():
    evaluator = SemanticEvaluator()
    result = evaluate_answer(
        question="q", generated_answer="a", reference_answer="r", evaluator=evaluator
    )
    assert result.judge_score is None
