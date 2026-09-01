#!/usr/bin/env python3
"""
Entry point for the LLM Performance & Cost Observatory.

Usage:
    python run_benchmark.py

By default runs in LOCAL_SIMULATION mode with zero API keys required.
Set USE_REAL_MODE=true and GOOGLE_API_KEY=... (see .env.example) to
optionally benchmark a real Gemini model instead of / alongside the
simulated local profiles.
"""

import logging
import random
import sys

from src import config
from src.benchmark import run_benchmark
from src.config import PricingConfig, ScoringWeights
from src.providers import build_local_profiles, GeminiProvider
from src.reporting import generate_all_reports

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("llm_observatory.main")


def main() -> None:
    random.seed(config.RANDOM_SEED)

    if config.USE_REAL_MODE and config.GOOGLE_API_KEY:
        mode = "REAL_GEMINI"
        logger.info("Real mode enabled: benchmarking Gemini alongside local baseline")
        providers = build_local_profiles()[:1] + [
            GeminiProvider(model_name="gemini-1.5-flash", api_key=config.GOOGLE_API_KEY)
        ]
    else:
        mode = "LOCAL_SIMULATION"
        if config.USE_REAL_MODE and not config.GOOGLE_API_KEY:
            logger.warning(
                "USE_REAL_MODE=true but GOOGLE_API_KEY is not set; "
                "falling back to LOCAL_SIMULATION mode."
            )
        providers = build_local_profiles()

    pricing_by_model = {
        "local-fast-lowcost": PricingConfig(input_cost_per_1m_tokens=0.10, output_cost_per_1m_tokens=0.30),
        "local-balanced": PricingConfig(input_cost_per_1m_tokens=0.50, output_cost_per_1m_tokens=1.50),
        "local-highquality": PricingConfig(input_cost_per_1m_tokens=2.00, output_cost_per_1m_tokens=6.00),
        "gemini-1.5-flash": PricingConfig(input_cost_per_1m_tokens=0.075, output_cost_per_1m_tokens=0.30),
    }

    weights = ScoringWeights()

    result = run_benchmark(providers=providers, pricing_by_model=pricing_by_model, weights=weights)
    generate_all_reports(result, mode=mode, weights=weights.as_dict())


if __name__ == "__main__":
    main()
