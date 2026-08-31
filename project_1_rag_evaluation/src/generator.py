"""Answer generation: a deterministic local mode plus an optional LLM mode.

MODE 1 (default, always available): a context-aware extractive generator
that selects and lightly stitches together the sentences from the
retrieved context most similar to the question. No network access or API
key required.

MODE 2 (optional): if `config.use_llm_generation` is True and a
GOOGLE_API_KEY is configured, Gemini is used to produce the final answer.
No API calls are ever made unless this is explicitly enabled -- and if the
call fails for any reason, the generator transparently falls back to the
local extractive mode so the pipeline never crashes.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from src.config import Config
from src.embeddings import EmbeddingModel, cosine_similarity
from src.logging_config import setup_logging
from src.models import RetrievalResult

logger = setup_logging()


def split_sentences(text: str) -> List[str]:
    """Very small, dependency-free sentence splitter."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def estimate_tokens(text: str) -> int:
    """Whitespace-based token count estimate.

    This is explicitly an estimate, NOT the exact tokenization used by any
    specific model or billing system.
    """
    if not text:
        return 0
    return max(1, len(text.split()))


class AnswerGenerator:
    """Generates an answer for a question given retrieved context chunks."""

    def __init__(self, config: Config, embedding_model: EmbeddingModel) -> None:
        self.config = config
        self.embedding_model = embedding_model
        self._gemini_client_checked = False
        self._gemini_available = False

    def generate(
        self, question: str, retrieved: List[RetrievalResult]
    ) -> Tuple[str, bool]:
        """Generate an answer.

        Returns (answer_text, used_llm) so callers/metrics can record which
        mode actually produced the answer.
        """
        if self.config.use_llm_generation and self.config.has_llm_access():
            llm_answer = self._try_llm_generate(question, retrieved)
            if llm_answer is not None:
                return llm_answer, True
            logger.warning("LLM generation failed or unavailable; using local mode.")

        return self._local_generate(question, retrieved), False

    # ------------------------------------------------------------------
    # MODE 1: local / deterministic
    # ------------------------------------------------------------------
    def _local_generate(self, question: str, retrieved: List[RetrievalResult]) -> str:
        if not retrieved:
            return "I could not find relevant information in the knowledge base."

        candidate_sentences: List[str] = []
        for result in retrieved:
            candidate_sentences.extend(split_sentences(result.text))

        if not candidate_sentences:
            return " ".join(r.text for r in retrieved)[:400]

        question_vec = self.embedding_model.encode([question])[0]
        sentence_vecs = self.embedding_model.encode(candidate_sentences)

        scored = [
            (cosine_similarity(question_vec, sentence_vecs[i]), candidate_sentences[i])
            for i in range(len(candidate_sentences))
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        top_n = min(3, len(scored))
        best_sentences = [s for _, s in scored[:top_n]]

        # Preserve original order of appearance for readability rather than
        # pure similarity order, which tends to read more naturally.
        ordered = [s for s in candidate_sentences if s in best_sentences]
        # De-duplicate while preserving order.
        seen = set()
        final_sentences = []
        for s in ordered:
            if s not in seen:
                final_sentences.append(s)
                seen.add(s)

        return " ".join(final_sentences)

    # ------------------------------------------------------------------
    # MODE 2: optional LLM (Gemini)
    # ------------------------------------------------------------------
    def _try_llm_generate(self, question: str, retrieved: List[RetrievalResult]):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            logger.warning("google-generativeai is not installed; skipping LLM generation.")
            return None

        try:
            genai.configure(api_key=self.config.google_api_key)
            model = genai.GenerativeModel(self.config.gemini_model_name)
            context = "\n\n".join(r.text for r in retrieved)
            prompt = (
                "Answer the question using ONLY the provided context. "
                "Be concise (2-4 sentences).\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            response = model.generate_content(prompt)
            text = getattr(response, "text", None)
            if not text:
                return None
            return text.strip()
        except Exception as exc:  # noqa: BLE001 - any API failure must not crash the pipeline
            logger.error("Gemini generation call failed: %s", exc)
            return None
