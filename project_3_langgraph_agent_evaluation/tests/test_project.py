"""Tests for calculator, knowledge search, routing, correctness, trajectory, regression."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import build_agent, run_agent, safe_calculate, knowledge_search
from src.metrics import numeric_match, tool_accuracy, trajectory_accuracy
from src.regression import compare


def test_calculator_operations():
    assert safe_calculate("15 + 27") == 42
    assert safe_calculate("100 - 45") == 55
    assert safe_calculate("12 * 8") == 96
    assert safe_calculate("144 / 12") == 12


def test_calculator_rejects_division_by_zero():
    try:
        safe_calculate("5 / 0")
        assert False, "expected ZeroDivisionError"
    except ZeroDivisionError:
        pass


def test_knowledge_search_matches_topic():
    result = knowledge_search("What is RAG used for?")
    assert "retrieval" in result.lower()

    result = knowledge_search("Why do models hallucinate?")
    assert "hallucinat" in result.lower()


def test_router_v1_calculator_case():
    app = build_agent(buggy=False)
    result = run_agent(app, "Evaluate 12 * 8")
    assert result["route"] == "calculator"
    assert result["path"] == ["router", "calculator", "answer"]


def test_router_v2_bug_misroutes_to_knowledge():
    app = build_agent(buggy=True)
    result = run_agent(app, "Evaluate 12 * 8")
    assert result["route"] == "knowledge"  # the intentional V2 bug


def test_router_direct_case():
    app = build_agent(buggy=False)
    result = run_agent(app, "Hello, how are you today?")
    assert result["route"] == "direct"


def test_numeric_answer_correctness():
    assert numeric_match("42", "42") == 1.0
    assert numeric_match("41", "42") == 0.0


def test_tool_accuracy():
    assert tool_accuracy("calculator", "calculator") is True
    assert tool_accuracy("calculator", "knowledge") is False


def test_trajectory_accuracy():
    path = ["router", "calculator", "answer", "evaluate"]
    assert trajectory_accuracy(path, path) is True
    assert trajectory_accuracy(path, ["router", "knowledge", "answer", "evaluate"]) is False


def test_regression_detection_flags_a_real_drop():
    v1 = {"answer_correctness": 1.0, "tool_accuracy": 1.0, "trajectory_accuracy": 1.0,
          "reliability": 1.0, "mean_latency": 0.002}
    v2 = {"answer_correctness": 0.8, "tool_accuracy": 0.8, "trajectory_accuracy": 0.8,
          "reliability": 0.8, "mean_latency": 0.002}
    report = compare(v1, v2)
    assert report["status"] == "REGRESSION DETECTED"


def test_regression_detection_allows_small_drop():
    v1 = {"answer_correctness": 1.0, "tool_accuracy": 1.0, "trajectory_accuracy": 1.0,
          "reliability": 1.0, "mean_latency": 0.002}
    v2 = {"answer_correctness": 0.98, "tool_accuracy": 1.0, "trajectory_accuracy": 1.0,
          "reliability": 1.0, "mean_latency": 0.002}
    report = compare(v1, v2)
    assert report["status"] == "NO REGRESSION"
