import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PricingConfig
from src.metrics import calculate_cost, cost_per_successful_answer, cost_per_quality_point
from src.providers import estimate_tokens


def test_calculate_cost_basic():
    pricing = PricingConfig(input_cost_per_1m_tokens=1.0, output_cost_per_1m_tokens=2.0)
    result = calculate_cost(1_000_000, 500_000, pricing)
    assert math_isclose(result["input_cost"], 1.0)
    assert math_isclose(result["output_cost"], 1.0)
    assert math_isclose(result["total_cost"], 2.0)


def test_calculate_cost_zero_tokens():
    pricing = PricingConfig(input_cost_per_1m_tokens=1.0, output_cost_per_1m_tokens=2.0)
    result = calculate_cost(0, 0, pricing)
    assert result["total_cost"] == 0.0


def test_cost_per_successful_answer():
    assert cost_per_successful_answer(10.0, 5) == 2.0


def test_cost_per_successful_answer_zero_successes():
    assert cost_per_successful_answer(10.0, 0) == 0.0


def test_cost_per_quality_point():
    assert cost_per_quality_point(1.0, 0.5) == 2.0


def test_cost_per_quality_point_zero_quality():
    assert cost_per_quality_point(1.0, 0.0) == float("inf")


def test_estimate_tokens_basic():
    assert estimate_tokens("hello world") == 2


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 1


def test_estimate_tokens_single_word():
    assert estimate_tokens("hello") == 1


def math_isclose(a, b, tol=1e-9):
    return abs(a - b) < tol
