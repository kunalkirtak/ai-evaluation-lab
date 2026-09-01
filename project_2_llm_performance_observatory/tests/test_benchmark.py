import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark import run_benchmark, aggregate_by_model, build_recommendation
from src.config import PricingConfig, ScoringWeights
from src.providers import LocalMockProvider
from src.models import BenchmarkRecord


def _tiny_providers():
    return [
        LocalMockProvider(
            model_name="test-fast",
            base_latency=0.001, latency_jitter=0.001,
            quality_center=0.6, quality_spread=0.05,
            failure_probability=0.0, avg_output_tokens=10,
            rng=random.Random(1),
        ),
        LocalMockProvider(
            model_name="test-good",
            base_latency=0.002, latency_jitter=0.001,
            quality_center=0.95, quality_spread=0.02,
            failure_probability=0.0, avg_output_tokens=20,
            rng=random.Random(2),
        ),
    ]


def test_run_benchmark_produces_records_for_all_providers_and_questions():
    providers = _tiny_providers()
    result = run_benchmark(providers=providers, weights=ScoringWeights())
    # 18 questions x 2 providers
    assert len(result["records"]) == result["test_case_count"] * result["provider_count"]


def test_run_benchmark_comparison_has_one_row_per_model():
    providers = _tiny_providers()
    result = run_benchmark(providers=providers, weights=ScoringWeights())
    model_names = {row["model"] for row in result["comparison"]}
    assert model_names == {"test-fast", "test-good"}


def test_run_benchmark_recommendation_present():
    providers = _tiny_providers()
    result = run_benchmark(providers=providers, weights=ScoringWeights())
    rec = result["recommendation"]
    for key in ("best_quality", "fastest", "cheapest", "most_reliable", "best_balanced"):
        assert key in rec


def test_aggregate_by_model_handles_empty_records():
    assert aggregate_by_model([], run_duration=1.0) == []


def test_build_recommendation_handles_empty_comparison():
    assert build_recommendation([], []) == {}


def test_run_benchmark_with_failures_included():
    providers = [
        LocalMockProvider(
            model_name="test-unreliable",
            base_latency=0.001, latency_jitter=0.001,
            quality_center=0.7, quality_spread=0.05,
            failure_probability=1.0, avg_output_tokens=10,
            rng=random.Random(3),
        )
    ]
    result = run_benchmark(providers=providers, weights=ScoringWeights())
    row = result["comparison"][0]
    assert row["success_rate"] == 0.0
    assert row["error_rate"] == 1.0
