"""EvaluationEngine: runs the full RAG pipeline per question and scores it."""

from __future__ import annotations

import time
from typing import List

from src.config import Config
from src.embeddings import EmbeddingModel
from src.generator import AnswerGenerator, estimate_tokens
from src.llm_judge import judge_with_llm
from src.logging_config import setup_logging
from src.metrics import compute_generation_metrics, compute_retrieval_metrics
from src.models import (
    EvalQuestion,
    EvaluationResult,
    LatencyBreakdown,
    TokenUsage,
)
from src.retriever import Retriever

logger = setup_logging()


def classify_failure(result: EvaluationResult, config: Config) -> str:
    """Assign a coarse failure category to an evaluation result.

    Categories (checked in priority order):
      - retrieval_failure: no relevant document was retrieved at all.
      - unsupported_answer: faithfulness is low (answer not grounded in context).
      - low_relevance: the answer does not address the question.
      - generation_failure: retrieval succeeded but answer accuracy is low.
      - correct_answer: none of the above triggered.
    """
    rm = result.retrieval_metrics
    gm = result.generation_metrics

    if rm.hit_rate_at_k == 0.0:
        return "retrieval_failure"
    if gm.faithfulness_score < config.faithfulness_support_threshold:
        return "unsupported_answer"
    if gm.answer_relevance < config.low_relevance_threshold:
        return "low_relevance"
    if gm.answer_accuracy < config.low_accuracy_threshold:
        return "generation_failure"
    return "correct_answer"


class EvaluationEngine:
    """Runs retrieval + generation + scoring for a set of questions."""

    def __init__(
        self,
        config: Config,
        retriever: Retriever,
        generator: AnswerGenerator,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.embedding_model = embedding_model

    def evaluate_question(self, eval_question: EvalQuestion) -> EvaluationResult:
        """Run the full pipeline for a single question and return a result."""
        try:
            retrieval_start = time.perf_counter()
            retrieved = self.retriever.retrieve(eval_question.question, top_k=self.config.top_k)
            retrieval_seconds = time.perf_counter() - retrieval_start
        except Exception as exc:  # noqa: BLE001
            logger.error("Retrieval failed for %s: %s", eval_question.id, exc)
            retrieved = []
            retrieval_seconds = 0.0

        try:
            generation_start = time.perf_counter()
            generated_answer, used_llm = self.generator.generate(eval_question.question, retrieved)
            generation_seconds = time.perf_counter() - generation_start
        except Exception as exc:  # noqa: BLE001
            logger.error("Generation failed for %s: %s", eval_question.id, exc)
            generated_answer = ""
            used_llm = False
            generation_seconds = 0.0

        evaluation_start = time.perf_counter()

        retrieved_ids = [r.doc_id for r in retrieved]
        retrieval_metrics = compute_retrieval_metrics(
            retrieved_ids, eval_question.relevant_document_ids
        )
        generation_metrics = compute_generation_metrics(
            question=eval_question.question,
            generated_answer=generated_answer,
            ground_truth=eval_question.ground_truth,
            context_chunks=[r.text for r in retrieved],
            embedding_model=self.embedding_model,
            config=self.config,
        )

        llm_judge_result = None
        if self.config.use_llm_judge:
            llm_judge_result = judge_with_llm(
                self.config,
                question=eval_question.question,
                context_chunks=[r.text for r in retrieved],
                ground_truth=eval_question.ground_truth,
                generated_answer=generated_answer,
            )

        evaluation_seconds = time.perf_counter() - evaluation_start

        input_text = eval_question.question + " ".join(r.text for r in retrieved)
        input_tokens = estimate_tokens(input_text)
        output_tokens = estimate_tokens(generated_answer)
        tokens = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        estimated_cost = self._estimate_cost(tokens, used_llm)

        result = EvaluationResult(
            question_id=eval_question.id,
            question=eval_question.question,
            ground_truth=eval_question.ground_truth,
            generated_answer=generated_answer,
            retrieved=retrieved,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            latency=LatencyBreakdown(
                retrieval_seconds=retrieval_seconds,
                generation_seconds=generation_seconds,
                evaluation_seconds=evaluation_seconds,
            ),
            tokens=tokens,
            estimated_cost_usd=estimated_cost,
            llm_judge=llm_judge_result,
            experiment_name=self.config.experiment_name,
        )
        result.failure_category = classify_failure(result, self.config)
        return result

    def evaluate_all(self, eval_questions: List[EvalQuestion]) -> List[EvaluationResult]:
        """Run evaluate_question for every question, logging progress."""
        results: List[EvaluationResult] = []
        for i, eq in enumerate(eval_questions, start=1):
            logger.info(
                "[%s] Evaluating %d/%d: %s", self.config.experiment_name, i, len(eval_questions), eq.id
            )
            results.append(self.evaluate_question(eq))
        return results

    def _estimate_cost(self, tokens: TokenUsage, used_llm: bool) -> float:
        """Estimate cost in USD. Only nonzero when an LLM call actually occurred.

        Uses configurable per-1M-token rates from Config; these are
        illustrative estimates, not verified production pricing.
        """
        if not used_llm:
            return 0.0
        input_cost = (tokens.input_tokens / 1_000_000) * self.config.input_cost_per_1m_tokens
        output_cost = (tokens.output_tokens / 1_000_000) * self.config.output_cost_per_1m_tokens
        return round(input_cost + output_cost, 8)
