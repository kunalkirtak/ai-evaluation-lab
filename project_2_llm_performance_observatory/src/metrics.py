"""
Pure statistical/economic metric functions used by the benchmark engine.

Kept as small, independently testable functions with no side effects so
they can be unit tested directly (see tests/test_metrics.py and
tests/test_cost.py).
"""

import math
from typing import List, Sequence, Dict, Any

from src.config import PricingConfig


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], pct: float) -> float:
    """
    Compute the pct-th percentile (0-100) of values using linear
    interpolation between closest ranks (same convention as numpy's
    default 'linear' method).
    """
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    k = (len(data) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(data[int(k)])
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return float(d0 + d1)


def latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(latencies) / len(latencies),
        "median": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "min": min(latencies),
        "max": max(latencies),
    }


# ---------------------------------------------------------------------------
# Tokens & Cost
# ---------------------------------------------------------------------------
def calculate_cost(input_tokens: int, output_tokens: int, pricing: PricingConfig) -> Dict[str, float]:
    """
    Compute input/output/total cost for a single request given a
    (configurable, example-only) pricing assumption expressed as USD per
    1,000,000 tokens.
    """
    input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_1m_tokens
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m_tokens
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not denominator:
        return default
    return numerator / denominator


def cost_per_successful_answer(total_cost: float, successful_requests: int) -> float:
    return safe_divide(total_cost, successful_requests, default=0.0)


def cost_per_quality_point(total_cost: float, mean_quality: float) -> float:
    """
    Project-defined metric: total cost divided by mean quality score.
    Lower is better (cheaper cost to achieve each unit of quality). This
    is NOT a standard industry metric; it exists to make the
    quality/cost tradeoff easy to compare across configurations.
    """
    return safe_divide(total_cost, mean_quality, default=float("inf"))


# ---------------------------------------------------------------------------
# Reliability & Throughput
# ---------------------------------------------------------------------------
def success_rate(successful: int, total: int) -> float:
    return safe_divide(successful, total, default=0.0)


def error_rate(failed: int, total: int) -> float:
    return safe_divide(failed, total, default=0.0)


def throughput_rps(total_requests: int, duration_seconds: float) -> float:
    return safe_divide(total_requests, duration_seconds, default=0.0)


# ---------------------------------------------------------------------------
# Composite production score
# ---------------------------------------------------------------------------
def normalize_min_max(value: float, min_val: float, max_val: float, invert: bool = False) -> float:
    """
    Normalize a value into [0, 1] given the observed min/max across all
    compared configurations. If invert=True, a lower raw value maps to a
    HIGHER normalized score (used for latency and cost, where lower is
    better).
    """
    if max_val == min_val:
        return 1.0
    norm = (value - min_val) / (max_val - min_val)
    norm = max(0.0, min(1.0, norm))
    return (1.0 - norm) if invert else norm


def composite_production_score(
    quality: float,
    reliability: float,
    normalized_speed: float,
    normalized_cost_efficiency: float,
    weights: Dict[str, float],
) -> float:
    """
    Project-defined composite score combining quality, reliability,
    speed, and cost efficiency. Higher is always better. Weights are
    fully configurable (src/config.py) and this is explicitly NOT an
    industry-standard formula - it is one reasonable way to combine
    multiple production concerns into a single ranking signal.
    """
    return (
        weights.get("quality", 0.5) * quality
        + weights.get("reliability", 0.2) * reliability
        + weights.get("latency", 0.15) * normalized_speed
        + weights.get("cost", 0.15) * normalized_cost_efficiency
    )


# ---------------------------------------------------------------------------
# Pareto efficiency
# ---------------------------------------------------------------------------
def pareto_efficient(candidates: List[Dict[str, Any]]) -> List[str]:
    """
    Identify Pareto-efficient configurations across three objectives:
      - quality (maximize)
      - mean_latency (minimize)
      - cost_per_request (minimize)

    Each candidate dict must contain: 'name', 'quality', 'mean_latency',
    'cost_per_request'.

    A configuration is Pareto-efficient if no other configuration is at
    least as good on ALL three objectives while being strictly better on
    at least one. Returns the list of Pareto-efficient configuration
    names.
    """
    names = []
    for i, a in enumerate(candidates):
        dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue
            at_least_as_good = (
                b["quality"] >= a["quality"]
                and b["mean_latency"] <= a["mean_latency"]
                and b["cost_per_request"] <= a["cost_per_request"]
            )
            strictly_better = (
                b["quality"] > a["quality"]
                or b["mean_latency"] < a["mean_latency"]
                or b["cost_per_request"] < a["cost_per_request"]
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            names.append(a["name"])
    return names
