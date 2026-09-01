"""
Quality evaluation for generated answers.

Default evaluation is fully local and does not require any external LLM:

  1. Semantic similarity: generated answer vs. reference answer, using
     sentence-transformers (all-MiniLM-L6-v2 by default).
  2. Answer relevance: question vs. generated answer, using the same
     embedding model.
  3. Correctness: a configurable threshold (QUALITY_THRESHOLD) is applied
     to the similarity score to classify an answer as correct/incorrect
     for this benchmark. This is a project-defined experiment setting.

An OPTIONAL LLM-as-judge can be enabled via config.ENABLE_LLM_JUDGE. It
is never required and the project works fully without it.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np

from src import config
from src.models import EvaluationResult

logger = logging.getLogger("llm_observatory.evaluator")

_embedding_model = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    raw = float(np.dot(a, b) / denom)  # in [-1, 1]
    # Map to [0, 1] for interpretability as a quality/relevance score.
    normalized = (raw + 1.0) / 2.0
    return max(0.0, min(1.0, normalized))


class SemanticEvaluator:
    """
    Wraps a sentence-transformers embedding model for semantic similarity
    and relevance scoring. The model is loaded lazily and cached so it is
    only downloaded/loaded once per process.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self._model = None

    def _load(self):
        global _embedding_model
        if _embedding_model is not None:
            self._model = _embedding_model
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            _embedding_model = self._model
        except Exception as exc:
            logger.warning(
                f"Could not load sentence-transformers model ({exc}); "
                "falling back to a lightweight bag-of-words similarity."
            )
            self._model = None

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self._load()
        if self._model is None:
            return None
        return self._model.encode(text, show_progress_bar=False)

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        emb_a = self._embed(text_a)
        emb_b = self._embed(text_b)
        if emb_a is not None and emb_b is not None:
            return _cosine_similarity(emb_a, emb_b)
        return _fallback_similarity(text_a, text_b)


def _fallback_similarity(text_a: str, text_b: str) -> float:
    """
    Simple Jaccard-style bag-of-words fallback used only if
    sentence-transformers is unavailable in the environment. Ensures the
    project still runs (with a cruder quality signal) with zero heavy
    dependencies.
    """
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def evaluate_answer(
    question: str,
    generated_answer: str,
    reference_answer: str,
    evaluator: SemanticEvaluator,
    judge_fn=None,
) -> EvaluationResult:
    """
    Run the default local quality evaluation, and optionally an LLM
    judge if judge_fn is provided and config.ENABLE_LLM_JUDGE is True.
    """
    quality_score = evaluator.similarity(generated_answer, reference_answer)
    relevance_score = evaluator.similarity(question, generated_answer)
    is_correct = quality_score >= config.QUALITY_THRESHOLD

    judge_score = None
    judge_raw = None
    if config.ENABLE_LLM_JUDGE and judge_fn is not None:
        try:
            judge_raw = judge_fn(question, generated_answer, reference_answer)
            judge_score = judge_raw.get("overall_score") if judge_raw else None
        except Exception as exc:
            logger.warning(f"LLM judge failed, continuing without it: {exc}")
            judge_score = None
            judge_raw = None

    return EvaluationResult(
        quality_score=round(quality_score, 4),
        relevance_score=round(relevance_score, 4),
        is_correct=is_correct,
        judge_score=judge_score,
        judge_raw=judge_raw,
    )


def optional_llm_judge(question: str, answer: str, reference: str) -> Dict[str, Any]:
    """
    OPTIONAL LLM-as-judge. Only called if ENABLE_LLM_JUDGE=true AND a
    real provider is configured. Requests a structured JSON verdict and
    validates it defensively. Any failure propagates to the caller,
    which records it and continues the benchmark without judge scores.
    """
    from src import config as cfg

    if not cfg.GOOGLE_API_KEY:
        raise RuntimeError("LLM judge requires GOOGLE_API_KEY to be configured")

    import google.generativeai as genai  # type: ignore
    import json as _json

    genai.configure(api_key=cfg.GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = (
        "You are grading an AI-generated answer. Respond with ONLY a JSON "
        "object with keys: correctness (0-1), relevance (0-1), "
        "completeness (0-1), overall_score (0-1).\n\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Generated answer: {answer}\n"
    )
    response = model.generate_content(prompt)
    text = (getattr(response, "text", "") or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    parsed = _json.loads(text)

    for key in ("correctness", "relevance", "completeness", "overall_score"):
        if key not in parsed:
            raise ValueError(f"LLM judge response missing key: {key}")
    return parsed
