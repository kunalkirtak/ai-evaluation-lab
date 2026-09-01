"""
Provider abstraction for the benchmarking engine.

The benchmark runner depends only on the LLMProvider interface, never on
a concrete provider like Gemini. This keeps real-model integration
strictly optional.

Providers implemented:
  1. LocalMockProvider - deterministic, simulated, no network/API key
     required. This is the DEFAULT mode and is what makes the whole
     project runnable with zero credentials.
  2. GeminiProvider - OPTIONAL real-model adapter. Only used if
     USE_REAL_MODE=true AND GOOGLE_API_KEY is configured. Any failure is
     caught, logged, and recorded as a failed GenerationResult; it never
     crashes the benchmark.
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

from src.models import GenerationResult

logger = logging.getLogger("llm_observatory.providers")


def estimate_tokens(text: str) -> int:
    """
    Lightweight local token estimator used ONLY for local/mock mode.

    This is NOT equivalent to a provider's real tokenizer and is clearly
    labeled as ESTIMATED_TOKENS everywhere it is used.
    """
    if not text:
        return 1
    return max(1, len(text.split()))


class LLMProvider(ABC):
    """Common interface every model adapter must implement."""

    model_name: str = "unknown"

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """Generate a response for the given prompt."""
        raise NotImplementedError


class LocalMockProvider(LLMProvider):
    """
    Deterministic simulated model adapter.

    IMPORTANT: This provider does NOT call any real LLM. It simulates
    latency, token usage, answer quality, and occasional failures so the
    benchmarking framework can be demonstrated end-to-end without an API
    key or network access. Results from this provider are NOT
    measurements of any real commercial model.

    Behavior is controlled by a small set of "profile" parameters so that
    multiple distinct simulated configurations (fast/cheap, balanced,
    high-quality/expensive) can be created from the same class.
    """

    def __init__(
        self,
        model_name: str,
        base_latency: float = 0.05,
        latency_jitter: float = 0.02,
        quality_center: float = 0.80,
        quality_spread: float = 0.10,
        failure_probability: float = 0.03,
        avg_output_tokens: int = 60,
        rng: Optional[random.Random] = None,
    ):
        self.model_name = model_name
        self.base_latency = base_latency
        self.latency_jitter = latency_jitter
        self.quality_center = quality_center
        self.quality_spread = quality_spread
        self.failure_probability = failure_probability
        self.avg_output_tokens = avg_output_tokens
        # Each provider gets its own RNG so results are reproducible and
        # independent of call order between providers.
        self._rng = rng or random.Random(hash(model_name) & 0xFFFFFFFF)

    def _simulated_answer(self, prompt: str, reference_hint: str) -> str:
        """
        Build a plausible-looking simulated answer. Quality is not
        actually driven by NLP here; it is a simulated profile used to
        exercise the benchmarking/evaluation pipeline realistically.
        """
        noise_words = [
            "essentially", "in practice", "generally speaking",
            "as a result", "for this reason", "notably",
        ]
        filler = self._rng.choice(noise_words)
        # Blend part of the reference with generic phrasing so semantic
        # similarity varies by simulated quality profile rather than
        # being either perfect or nonsensical.
        keep_ratio = max(0.15, min(0.95, self._rng.gauss(self.quality_center, self.quality_spread)))
        words = reference_hint.split()
        keep_n = max(3, int(len(words) * keep_ratio))
        kept = " ".join(words[:keep_n])
        return f"{kept} {filler}, this addresses the question about: {prompt[:40]}"

    def generate(self, prompt: str, reference_hint: str = "") -> GenerationResult:
        start = time.perf_counter()

        # Simulate occasional failures (timeouts / provider errors).
        if self._rng.random() < self.failure_probability:
            elapsed = time.perf_counter() - start
            simulated_latency = max(0.01, self.base_latency + self._rng.uniform(0, self.latency_jitter))
            time.sleep(min(simulated_latency, 0.05))
            logger.warning(f"Simulated request failure | model={self.model_name}")
            return GenerationResult(
                answer="",
                input_tokens=estimate_tokens(prompt),
                output_tokens=0,
                total_tokens=estimate_tokens(prompt),
                latency_seconds=time.perf_counter() - start,
                success=False,
                model_name=self.model_name,
                error="simulated_provider_error",
                token_source="ESTIMATED_TOKENS",
            )

        simulated_latency = max(0.005, self.base_latency + self._rng.uniform(0, self.latency_jitter))
        # Sleep briefly (capped) so the demo runs fast while still
        # producing non-trivial, distinguishable latency numbers.
        time.sleep(min(simulated_latency, 0.05))

        answer = self._simulated_answer(prompt, reference_hint or prompt)
        input_tokens = estimate_tokens(prompt)
        output_tokens = max(1, int(self._rng.gauss(self.avg_output_tokens, self.avg_output_tokens * 0.2)))

        return GenerationResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_seconds=time.perf_counter() - start,
            success=True,
            model_name=self.model_name,
            error=None,
            token_source="ESTIMATED_TOKENS",
        )


class GeminiProvider(LLMProvider):
    """
    OPTIONAL real-model adapter using Google's Gemini API.

    This provider is never instantiated unless the caller explicitly
    enables real mode AND a GOOGLE_API_KEY is present. Import of the
    google-generativeai SDK is deferred to __init__ so that the rest of
    the project works even if the package is not installed.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None
        try:
            import google.generativeai as genai  # type: ignore

            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required for GeminiProvider")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(model_name)
        except Exception as exc:  # pragma: no cover - exercised only in real mode
            logger.warning(f"GeminiProvider initialization failed: {exc}")
            self._client = None
            self._init_error = str(exc)

    def generate(self, prompt: str) -> GenerationResult:
        start = time.perf_counter()

        if self._client is None:
            return GenerationResult(
                answer="",
                input_tokens=estimate_tokens(prompt),
                output_tokens=0,
                total_tokens=estimate_tokens(prompt),
                latency_seconds=time.perf_counter() - start,
                success=False,
                model_name=self.model_name,
                error=getattr(self, "_init_error", "provider_not_initialized"),
                token_source="ESTIMATED_TOKENS",
            )

        try:
            response = self._client.generate_content(prompt)
            latency = time.perf_counter() - start
            answer = getattr(response, "text", "") or ""

            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_token_count", estimate_tokens(prompt))
                output_tokens = getattr(usage, "candidates_token_count", estimate_tokens(answer))
                token_source = "EXACT_PROVIDER_TOKENS"
            else:
                input_tokens = estimate_tokens(prompt)
                output_tokens = estimate_tokens(answer)
                token_source = "ESTIMATED_TOKENS"

            return GenerationResult(
                answer=answer,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                total_tokens=int(input_tokens) + int(output_tokens),
                latency_seconds=latency,
                success=True,
                model_name=self.model_name,
                error=None,
                token_source=token_source,
            )
        except Exception as exc:
            logger.warning(f"Gemini API request failed | model={self.model_name} | error={exc}")
            return GenerationResult(
                answer="",
                input_tokens=estimate_tokens(prompt),
                output_tokens=0,
                total_tokens=estimate_tokens(prompt),
                latency_seconds=time.perf_counter() - start,
                success=False,
                model_name=self.model_name,
                error=str(exc),
                token_source="ESTIMATED_TOKENS",
            )


def build_local_profiles() -> "list[LocalMockProvider]":
    """
    Create the three default simulated benchmark configurations described
    in the README: fast/low-cost, balanced, and high-quality/higher-cost.

    These are simulated benchmark profiles and are NOT measurements of
    any real commercial model.
    """
    return [
        LocalMockProvider(
            model_name="local-fast-lowcost",
            base_latency=0.02,
            latency_jitter=0.01,
            quality_center=0.65,
            quality_spread=0.12,
            failure_probability=0.05,
            avg_output_tokens=35,
            rng=random.Random(101),
        ),
        LocalMockProvider(
            model_name="local-balanced",
            base_latency=0.05,
            latency_jitter=0.02,
            quality_center=0.80,
            quality_spread=0.08,
            failure_probability=0.03,
            avg_output_tokens=60,
            rng=random.Random(202),
        ),
        LocalMockProvider(
            model_name="local-highquality",
            base_latency=0.10,
            latency_jitter=0.03,
            quality_center=0.92,
            quality_spread=0.05,
            failure_probability=0.01,
            avg_output_tokens=95,
            rng=random.Random(303),
        ),
    ]
