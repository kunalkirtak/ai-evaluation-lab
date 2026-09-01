"""
Shared data structures used across the benchmarking pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class TestCase:
    """A single benchmark question."""

    id: str
    question: str
    reference_answer: str
    category: str


@dataclass
class GenerationResult:
    """
    Standard result returned by any LLMProvider.generate() call.

    token_source distinguishes EXACT_PROVIDER_TOKENS (real usage reported
    by a provider API) from ESTIMATED_TOKENS (a lightweight local
    heuristic used in mock/local mode).
    """

    answer: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    success: bool
    model_name: str
    error: Optional[str] = None
    token_source: str = "ESTIMATED_TOKENS"


@dataclass
class EvaluationResult:
    """Quality evaluation attached to one generation."""

    quality_score: float
    relevance_score: float
    is_correct: bool
    judge_score: Optional[float] = None
    judge_raw: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkRecord:
    """One fully-processed benchmark request: generation + evaluation."""

    model_name: str
    question_id: str
    question: str
    category: str
    reference_answer: str
    generated_answer: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_source: str
    latency_seconds: float
    success: bool
    error: Optional[str]
    quality_score: float
    relevance_score: float
    is_correct: bool
    input_cost: float
    output_cost: float
    total_cost: float
    failure_category: str
