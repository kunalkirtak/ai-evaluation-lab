import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics import (
    percentile,
    latency_stats,
    success_rate,
    error_rate,
    throughput_rps,
    normalize_min_max,
    pareto_efficient,
    composite_production_score,
    safe_divide,
)


def test_percentile_median_odd():
    assert percentile([1, 2, 3], 50) == 2


def test_percentile_p95_matches_expected_range():
    values = list(range(1, 101))  # 1..100
    p95 = percentile(values, 95)
    assert 94 <= p95 <= 96


def test_percentile_empty():
    assert percentile([], 50) == 0.0


def test_latency_stats_basic():
    stats = latency_stats([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0


def test_success_rate():
    assert success_rate(8, 10) == 0.8


def test_success_rate_zero_total():
    assert success_rate(0, 0) == 0.0


def test_error_rate():
    assert error_rate(2, 10) == 0.2


def test_throughput_rps():
    assert throughput_rps(50, 10) == 5.0


def test_throughput_rps_zero_duration():
    assert throughput_rps(50, 0) == 0.0


def test_safe_divide_zero_denominator():
    assert safe_divide(5, 0, default=-1) == -1


def test_normalize_min_max_basic():
    assert normalize_min_max(5, 0, 10) == 0.5


def test_normalize_min_max_inverted():
    # Lower raw value -> higher normalized score when inverted (e.g. latency/cost).
    low = normalize_min_max(0, 0, 10, invert=True)
    high = normalize_min_max(10, 0, 10, invert=True)
    assert low > high
    assert low == 1.0
    assert high == 0.0


def test_normalize_min_max_equal_bounds():
    assert normalize_min_max(5, 5, 5) == 1.0


def test_composite_production_score_higher_is_better():
    weights = {"quality": 0.5, "reliability": 0.2, "latency": 0.15, "cost": 0.15}
    good = composite_production_score(0.9, 0.95, 0.9, 0.9, weights)
    bad = composite_production_score(0.3, 0.5, 0.2, 0.2, weights)
    assert good > bad


def test_pareto_efficient_identifies_non_dominated():
    candidates = [
        {"name": "cheap_fast_low_quality", "quality": 0.5, "mean_latency": 0.01, "cost_per_request": 0.001},
        {"name": "expensive_slow_high_quality", "quality": 0.95, "mean_latency": 0.2, "cost_per_request": 0.05},
        {"name": "dominated", "quality": 0.4, "mean_latency": 0.3, "cost_per_request": 0.06},
    ]
    result = pareto_efficient(candidates)
    assert "cheap_fast_low_quality" in result
    assert "expensive_slow_high_quality" in result
    assert "dominated" not in result


def test_pareto_efficient_single_candidate():
    candidates = [{"name": "only_one", "quality": 0.5, "mean_latency": 0.1, "cost_per_request": 0.01}]
    assert pareto_efficient(candidates) == ["only_one"]
