"""Compare Agent V1 vs Agent V2 aggregate metrics and detect regressions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

MAX_DROP = 0.05

# 0-1 scaled metrics where (v1 - v2) > MAX_DROP counts as a regression.
DROP_METRICS = ["answer_correctness", "tool_accuracy", "trajectory_accuracy", "reliability"]
ALL_METRICS = DROP_METRICS + ["mean_latency"]


def compare(v1_metrics: Dict[str, float], v2_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Compute per-metric deltas and decide REGRESSION DETECTED / NO REGRESSION."""
    rows = []
    regressed = []

    for key in ALL_METRICS:
        v1_val, v2_val = v1_metrics[key], v2_metrics[key]
        delta = v2_val - v1_val
        flagged = key in DROP_METRICS and (v1_val - v2_val) > MAX_DROP
        if flagged:
            regressed.append(key)
        rows.append(
            {
                "metric": key,
                "v1": round(v1_val, 4),
                "v2": round(v2_val, 4),
                "delta": round(delta, 4),
                "regressed": flagged,
            }
        )

    status = "REGRESSION DETECTED" if regressed else "NO REGRESSION"
    reason = (
        f"metric(s) dropped by more than {MAX_DROP}: {', '.join(regressed)}"
        if regressed
        else f"no metric dropped by more than {MAX_DROP}"
    )

    return {
        "status": status,
        "reason": reason,
        "max_drop_threshold": MAX_DROP,
        "v1_metrics": v1_metrics,
        "v2_metrics": v2_metrics,
        "comparison": rows,
    }


def save_report(results_dir: str, report: Dict[str, Any]) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report["comparison"]).to_csv(out / "regression_comparison.csv", index=False)
    with open(out / "regression_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
