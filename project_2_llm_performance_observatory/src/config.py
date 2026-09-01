"""
Configuration for the LLM Performance & Cost Observatory.

All values here are experiment / benchmark assumptions, not universal
truths. Pricing, weights, and thresholds are configurable and should be
adjusted to match your own workload and the real pricing of any provider
you evaluate.

Environment variables (see .env.example) can override the defaults below.
"""

import os
from dataclasses import dataclass, field
from typing import Dict


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------
# USE_REAL_MODE=false by default. The project MUST run fully without any
# API key in local simulation mode.
USE_REAL_MODE: bool = _env_bool("USE_REAL_MODE", False)

# Optional observability integration. Never required to run the benchmark.
ENABLE_LANGSMITH: bool = _env_bool("ENABLE_LANGSMITH", False)
LANGSMITH_TRACING: bool = _env_bool("LANGSMITH_TRACING", False)

# Optional LLM-as-judge. Never required.
ENABLE_LLM_JUDGE: bool = _env_bool("ENABLE_LLM_JUDGE", False)

GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------
# Score (0-1) above which a generated answer is treated as "correct enough"
# for this benchmark. This is a project-defined experiment setting, not an
# industry standard.
QUALITY_THRESHOLD: float = _env_float("QUALITY_THRESHOLD", 0.70)
RELEVANCE_THRESHOLD: float = _env_float("RELEVANCE_THRESHOLD", 0.60)

# sentence-transformers model used for semantic similarity / relevance.
EMBEDDING_MODEL_NAME: str = os.environ.get(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Pricing assumptions (example only - NOT guaranteed to match any real
# commercial provider's current pricing). Prices are USD per 1,000,000
# tokens. Override per-model below or via ModelConfig.
# ---------------------------------------------------------------------------
DEFAULT_INPUT_COST_PER_1M: float = _env_float("DEFAULT_INPUT_COST_PER_1M", 0.50)
DEFAULT_OUTPUT_COST_PER_1M: float = _env_float("DEFAULT_OUTPUT_COST_PER_1M", 1.50)

# ---------------------------------------------------------------------------
# Composite "production score" weights.
# This is a project-defined decision framework, NOT an industry-standard
# formula. Weights must sum to 1.0 for the score to stay in a comparable
# [0, 1] range, but the code does not hard-enforce this.
# ---------------------------------------------------------------------------
QUALITY_WEIGHT: float = _env_float("QUALITY_WEIGHT", 0.50)
RELIABILITY_WEIGHT: float = _env_float("RELIABILITY_WEIGHT", 0.20)
LATENCY_WEIGHT: float = _env_float("LATENCY_WEIGHT", 0.15)
COST_WEIGHT: float = _env_float("COST_WEIGHT", 0.15)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "benchmark_dataset.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


@dataclass
class PricingConfig:
    """Example / configurable pricing assumption for a single model."""

    input_cost_per_1m_tokens: float = DEFAULT_INPUT_COST_PER_1M
    output_cost_per_1m_tokens: float = DEFAULT_OUTPUT_COST_PER_1M


@dataclass
class ScoringWeights:
    quality: float = QUALITY_WEIGHT
    reliability: float = RELIABILITY_WEIGHT
    latency: float = LATENCY_WEIGHT
    cost: float = COST_WEIGHT

    def as_dict(self) -> Dict[str, float]:
        return {
            "quality": self.quality,
            "reliability": self.reliability,
            "latency": self.latency,
            "cost": self.cost,
        }


def ensure_result_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
