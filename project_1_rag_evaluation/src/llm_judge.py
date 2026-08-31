"""Optional Gemini-based LLM-as-judge evaluator.

This module is entirely optional. It is only invoked when
`config.use_llm_judge` is True AND a GOOGLE_API_KEY is configured. If the
`google-generativeai` package is missing, the API call fails, or the
response cannot be parsed as valid JSON, this module logs the problem and
returns an LLMJudgeResult with `available=False`, so the rest of the
evaluation pipeline is completely unaffected.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from src.config import Config
from src.logging_config import setup_logging
from src.models import LLMJudgeResult

logger = setup_logging()

_JUDGE_PROMPT_TEMPLATE = """You are evaluating the output of a RAG system. \
Respond with ONLY a JSON object, no markdown, no commentary.

Question: {question}
Retrieved context: {context}
Ground truth answer: {ground_truth}
Generated answer: {generated_answer}

Score each of the following from 0.0 to 1.0:
- "correctness": does the generated answer match the ground truth meaning?
- "faithfulness": is the generated answer supported by the retrieved context?
- "relevance": does the generated answer actually address the question?

Return JSON exactly in this shape:
{{"correctness": <float>, "faithfulness": <float>, "relevance": <float>, "rationale": "<one sentence>"}}
"""


def _extract_json(raw_text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from an LLM response string."""
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```(json)?", "", raw_text).strip()
    raw_text = re.sub(r"```$", "", raw_text).strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _validate_scores(data: dict) -> Optional[LLMJudgeResult]:
    required = ["correctness", "faithfulness", "relevance"]
    if not all(key in data for key in required):
        return None
    try:
        scores = {key: float(data[key]) for key in required}
    except (TypeError, ValueError):
        return None
    for key, value in scores.items():
        if not (0.0 <= value <= 1.0):
            scores[key] = max(0.0, min(1.0, value))
    return LLMJudgeResult(
        correctness=scores["correctness"],
        faithfulness=scores["faithfulness"],
        relevance=scores["relevance"],
        rationale=str(data.get("rationale", ""))[:500],
        available=True,
    )


def judge_with_llm(
    config: Config,
    question: str,
    context_chunks: List[str],
    ground_truth: str,
    generated_answer: str,
) -> LLMJudgeResult:
    """Run the optional Gemini LLM-as-judge evaluation for one example.

    Always returns an LLMJudgeResult. `available` is False whenever the
    judge could not run or produce valid structured output, in which case
    the deterministic metrics remain the source of truth for that example.
    """
    if not config.use_llm_judge:
        return LLMJudgeResult(available=False, error="llm_judge_disabled")
    if not config.has_llm_access():
        return LLMJudgeResult(available=False, error="no_api_key_configured")

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        logger.warning("google-generativeai not installed; skipping LLM judge.")
        return LLMJudgeResult(available=False, error="google-generativeai_not_installed")

    try:
        genai.configure(api_key=config.google_api_key)
        model = genai.GenerativeModel(config.gemini_model_name)
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            context="\n".join(context_chunks),
            ground_truth=ground_truth,
            generated_answer=generated_answer,
        )
        response = model.generate_content(prompt)
        raw_text = getattr(response, "text", "") or ""
    except Exception as exc:  # noqa: BLE001 - any API failure must not crash the pipeline
        logger.error("LLM judge call failed: %s", exc)
        return LLMJudgeResult(available=False, error=str(exc))

    parsed = _extract_json(raw_text)
    if parsed is None:
        logger.warning("LLM judge returned unparseable output; ignoring.")
        return LLMJudgeResult(available=False, error="unparseable_response")

    validated = _validate_scores(parsed)
    if validated is None:
        logger.warning("LLM judge JSON failed validation; ignoring.")
        return LLMJudgeResult(available=False, error="invalid_json_schema")

    return validated
