"""Runs an agent version over the dataset and produces per-case + aggregate results."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.agent import build_agent, run_agent
from src.metrics import answer_correctness, latency_stats, reliability, tool_accuracy, trajectory_accuracy

logger = logging.getLogger(__name__)

ANSWER_SCORE_THRESHOLD = 0.5  # minimum answer_score counted as a "successful" case


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_agent(version: str, buggy: bool, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run every case through one agent version and collect per-case + aggregate metrics."""
    app = build_agent(buggy=buggy)
    rows = []
    successes = 0

    for case in dataset:
        start = time.perf_counter()
        failure_reason = ""
        try:
            result = run_agent(app, case["input"])
            actual_tool = result["route"]
            actual_answer = result["answer"]
            actual_path = result["path"] + ["evaluate"]  # evaluation step appended by the harness
            ran_ok = True
        except Exception as exc:  # noqa: BLE001 - agent failures are recorded, not raised
            logger.warning("Case %s raised an exception: %s", case["id"], exc)
            actual_tool, actual_answer, actual_path = "error", "", ["router", "error"]
            ran_ok = False
            failure_reason = f"exception: {exc}"

        latency = time.perf_counter() - start

        if ran_ok:
            score = answer_correctness(actual_answer, case["reference_answer"], case["expected_tool"])
            t_ok = tool_accuracy(case["expected_tool"], actual_tool)
            p_ok = trajectory_accuracy(case["expected_path"], actual_path)
            if not t_ok:
                failure_reason = "wrong tool"
            elif not p_ok:
                failure_reason = "wrong trajectory"
            elif score < ANSWER_SCORE_THRESHOLD:
                failure_reason = "low answer score"
            success = t_ok and p_ok and score >= ANSWER_SCORE_THRESHOLD
        else:
            score, t_ok, p_ok, success = 0.0, False, False, False

        successes += int(success)
        rows.append(
            {
                "agent_version": version,
                "case_id": case["id"],
                "input": case["input"],
                "expected_tool": case["expected_tool"],
                "actual_tool": actual_tool,
                "expected_path": " > ".join(case["expected_path"]),
                "actual_path": " > ".join(actual_path),
                "answer_score": round(score, 4),
                "tool_correct": t_ok,
                "trajectory_correct": p_ok,
                "latency": round(latency, 6),
                "success": success,
                "failure_reason": failure_reason,
            }
        )

    df = pd.DataFrame(rows)
    lat_stats = latency_stats(df["latency"].tolist())
    metrics = {
        "answer_correctness": float(df["answer_score"].mean()),
        "tool_accuracy": float(df["tool_correct"].mean()),
        "trajectory_accuracy": float(df["trajectory_correct"].mean()),
        "reliability": reliability(successes, len(dataset)),
        "mean_latency": lat_stats["mean"],
        "median_latency": lat_stats["median"],
        "p95_latency": lat_stats["p95"],
    }
    return {"rows": df, "metrics": metrics}


def save_results(results_dir: str, v1: Dict[str, Any], v2: Dict[str, Any]) -> None:
    """Write evaluation_results.csv (all cases, both versions) and error_analysis.csv."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = pd.concat([v1["rows"], v2["rows"]], ignore_index=True)
    all_rows.to_csv(out / "evaluation_results.csv", index=False)

    error_cols = [
        "agent_version",
        "case_id",
        "input",
        "expected_tool",
        "actual_tool",
        "expected_path",
        "actual_path",
        "answer_score",
        "latency",
        "failure_reason",
    ]
    errors = all_rows[~all_rows["success"]][error_cols]
    errors.to_csv(out / "error_analysis.csv", index=False)
