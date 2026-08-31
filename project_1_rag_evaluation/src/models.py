"""Typed data structures shared across the RAG evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A raw source document loaded from data/documents/."""

    doc_id: str
    text: str
    source_path: str = ""


@dataclass
class Chunk:
    """A chunk produced from a Document, ready to be embedded."""

    chunk_id: str
    doc_id: str
    text: str


@dataclass
class EvalQuestion:
    """A single evaluation example."""

    id: str
    question: str
    ground_truth: str
    relevant_document_ids: List[str]


@dataclass
class RetrievalResult:
    """One retrieved chunk with its similarity score and rank."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LatencyBreakdown:
    retrieval_seconds: float
    generation_seconds: float
    evaluation_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.retrieval_seconds
            + self.generation_seconds
            + self.evaluation_seconds
        )


@dataclass
class GenerationMetrics:
    answer_accuracy: float
    answer_relevance: float
    faithfulness_score: float
    hallucination_rate: float
    unsupported_sentence_count: int
    total_sentence_count: int


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate_at_k: float


@dataclass
class LLMJudgeResult:
    """Structured output of the optional Gemini LLM-as-judge evaluator."""

    correctness: Optional[float] = None
    faithfulness: Optional[float] = None
    relevance: Optional[float] = None
    rationale: str = ""
    available: bool = False
    error: str = ""


@dataclass
class EvaluationResult:
    """Full evaluation record for a single question, for one experiment."""

    question_id: str
    question: str
    ground_truth: str
    generated_answer: str
    retrieved: List[RetrievalResult]
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    latency: LatencyBreakdown
    tokens: TokenUsage
    estimated_cost_usd: float
    llm_judge: Optional[LLMJudgeResult] = None
    failure_category: str = "unclassified"
    experiment_name: str = "default"

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten this result into a single-level dict for CSV export."""
        retrieved_ids = [r.doc_id for r in self.retrieved]
        row: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "generated_answer": self.generated_answer,
            "retrieved_doc_ids": ";".join(retrieved_ids),
            "top_retrieval_score": self.retrieved[0].score if self.retrieved else 0.0,
            "precision_at_k": self.retrieval_metrics.precision_at_k,
            "recall_at_k": self.retrieval_metrics.recall_at_k,
            "mrr": self.retrieval_metrics.mrr,
            "hit_rate_at_k": self.retrieval_metrics.hit_rate_at_k,
            "answer_accuracy": self.generation_metrics.answer_accuracy,
            "answer_relevance": self.generation_metrics.answer_relevance,
            "faithfulness_score": self.generation_metrics.faithfulness_score,
            "hallucination_rate": self.generation_metrics.hallucination_rate,
            "unsupported_sentence_count": self.generation_metrics.unsupported_sentence_count,
            "retrieval_latency_s": self.latency.retrieval_seconds,
            "generation_latency_s": self.latency.generation_seconds,
            "evaluation_latency_s": self.latency.evaluation_seconds,
            "total_latency_s": self.latency.total_seconds,
            "input_tokens": self.tokens.input_tokens,
            "output_tokens": self.tokens.output_tokens,
            "total_tokens": self.tokens.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "failure_category": self.failure_category,
        }
        if self.llm_judge is not None and self.llm_judge.available:
            row["llm_judge_correctness"] = self.llm_judge.correctness
            row["llm_judge_faithfulness"] = self.llm_judge.faithfulness
            row["llm_judge_relevance"] = self.llm_judge.relevance
        return row
