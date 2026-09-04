"""Evaluation metrics: answer correctness, tool/trajectory accuracy, reliability, latency."""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_model = None  # lazily-loaded SentenceTransformer, shared across calls


def _get_model():
    """Load the SentenceTransformer model once and cache it (downloaded on first use)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def semantic_similarity(actual: str, expected: str) -> float:
    """Cosine similarity between two texts using MiniLM embeddings, clamped to [0, 1]."""
    model = _get_model()
    emb = model.encode([actual, expected], normalize_embeddings=True)
    score = float(np.dot(emb[0], emb[1]))
    return max(0.0, min(1.0, score))


def _extract_number(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def numeric_match(actual: str, expected: str, tol: float = 1e-6) -> float:
    """1.0 if the numeric value in `actual` matches `expected` within tolerance, else 0.0."""
    a, e = _extract_number(actual), _extract_number(expected)
    if a is None or e is None:
        return 0.0
    return 1.0 if abs(a - e) <= tol else 0.0


def answer_correctness(actual: str, expected: str, expected_tool: str) -> float:
    """Deterministic numeric comparison for calculator answers, semantic similarity otherwise."""
    if expected_tool == "calculator":
        return numeric_match(actual, expected)
    return semantic_similarity(actual, expected)


def tool_accuracy(expected_tool: str, actual_tool: str) -> bool:
    return expected_tool == actual_tool


def trajectory_accuracy(expected_path: Sequence[str], actual_path: Sequence[str]) -> bool:
    return list(expected_path) == list(actual_path)


def reliability(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def latency_stats(latencies: List[float]) -> dict:
    """Mean, median and p95 latency (seconds) via numpy."""
    if not latencies:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    arr = np.array(latencies, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }
