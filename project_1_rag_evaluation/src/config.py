"""
Central configuration for the RAG Evaluation & Benchmarking System.

All tunables live here so that experiments (e.g. different top_k values)
can be run by constructing multiple Config instances rather than editing
code throughout the project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load a local .env file if python-dotenv is available. This is optional:
# the project must work perfectly well without any .env file at all.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is an optional convenience
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EVAL_DATASET_PATH = DATA_DIR / "evaluation_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RANDOM_SEED = 42


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class Config:
    """Runtime configuration for a single evaluation run/experiment."""

    # --- Retrieval ---
    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 3
    chunk_size_words: int = 60
    chunk_overlap_words: int = 15

    # --- Generation ---
    # If True and GOOGLE_API_KEY is set, the generator will call Gemini.
    # Defaults to False so the project runs fully offline out of the box.
    use_llm_generation: bool = field(
        default_factory=lambda: _env_bool("USE_LLM_GENERATION", False)
    )
    # If True and GOOGLE_API_KEY is set, an additional LLM-as-judge pass
    # is run on top of the deterministic metrics. Off by default.
    use_llm_judge: bool = field(
        default_factory=lambda: _env_bool("USE_LLM_JUDGE", False)
    )
    gemini_model_name: str = "gemini-1.5-flash"
    google_api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY", "")
    )

    # --- Evaluation thresholds ---
    # Minimum cosine similarity for an answer sentence to be considered
    # "supported" by the retrieved context during faithfulness scoring.
    faithfulness_support_threshold: float = 0.5
    # Below this accuracy score, an example is flagged in error analysis.
    low_accuracy_threshold: float = 0.55
    low_relevance_threshold: float = 0.55

    # --- Cost estimation (configurable estimates, NOT official pricing) ---
    input_cost_per_1m_tokens: float = field(
        default_factory=lambda: _env_float("INPUT_COST_PER_1M_TOKENS", 0.075)
    )
    output_cost_per_1m_tokens: float = field(
        default_factory=lambda: _env_float("OUTPUT_COST_PER_1M_TOKENS", 0.30)
    )

    # --- Misc ---
    random_seed: int = RANDOM_SEED
    experiment_name: str = "default"

    def has_llm_access(self) -> bool:
        """Whether a Google API key is configured at all."""
        return bool(self.google_api_key.strip())


def ensure_result_dirs() -> None:
    """Create results/ and results/plots/ if they do not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
