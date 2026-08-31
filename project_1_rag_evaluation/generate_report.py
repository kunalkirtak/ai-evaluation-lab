#!/usr/bin/env python
"""Standalone reporting script.

Loads previously generated results/evaluation_results.csv, recomputes
summary statistics, prints a clean terminal report, regenerates plots,
and rewrites results/evaluation_summary.json.

Usage:
    python run_evaluation.py     # first, to produce evaluation_results.csv
    python generate_report.py    # then, to (re)generate the report/plots
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, PLOTS_DIR, ensure_result_dirs
from src.logging_config import setup_logging
from src.reporting import print_terminal_report, save_summary_json

logger = setup_logging()

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


def _percentile(values, pct: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def summary_from_csv(df: pd.DataFrame) -> dict:
    latencies = df["total_latency_s"].tolist()
    return {
        "num_test_cases": int(len(df)),
        "mean_precision_at_k": float(df["precision_at_k"].mean()),
        "mean_recall_at_k": float(df["recall_at_k"].mean()),
        "mean_mrr": float(df["mrr"].mean()),
        "mean_hit_rate_at_k": float(df["hit_rate_at_k"].mean()),
        "mean_answer_accuracy": float(df["answer_accuracy"].mean()),
        "mean_faithfulness": float(df["faithfulness_score"].mean()),
        "mean_answer_relevance": float(df["answer_relevance"].mean()),
        "mean_hallucination_rate": float(df["hallucination_rate"].mean()),
        "mean_latency_seconds": float(np.mean(latencies)),
        "median_latency_seconds": float(np.median(latencies)),
        "p95_latency_seconds": _percentile(latencies, 95),
        "total_estimated_tokens": int(df["total_tokens"].sum()),
        "estimated_total_cost_usd": float(df["estimated_cost_usd"].sum()),
    }


def regenerate_plots_from_df(df: pd.DataFrame, plots_dir: Path) -> None:
    if not _HAS_MPL:
        logger.warning("matplotlib not available; skipping plot generation.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Retrieval metrics
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ["precision_at_k", "recall_at_k", "mrr", "hit_rate_at_k"]
    labels = ["Precision@K", "Recall@K", "MRR", "Hit Rate@K"]
    means = [df[m].mean() for m in metrics]
    bars = ax.bar(labels, means, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval Metrics Comparison")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(plots_dir / "retrieval_metrics.png", dpi=150)
    plt.close(fig)

    # Generation metrics
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ["answer_accuracy", "faithfulness_score", "answer_relevance", "hallucination_rate"]
    labels = ["Answer Accuracy", "Faithfulness", "Answer Relevance", "Hallucination Rate"]
    means = [df[m].mean() for m in metrics]
    colors = ["#55A868", "#55A868", "#55A868", "#C44E52"]
    bars = ax.bar(labels, means, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("Generation Quality Metrics")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(plots_dir / "generation_metrics.png", dpi=150)
    plt.close(fig)

    # Latency distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["total_latency_s"], bins=min(10, max(3, len(df) // 2)), color="#8172B2", edgecolor="white")
    ax.set_xlabel("Total Latency (seconds)")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Latency Distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "latency_distribution.png", dpi=150)
    plt.close(fig)

    # Cost/token summary
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    token_totals = [df["input_tokens"].sum(), df["output_tokens"].sum()]
    axes[0].bar(["Input Tokens", "Output Tokens"], token_totals, color="#CCB974")
    axes[0].set_title("Estimated Token Usage")
    total_cost = df["estimated_cost_usd"].sum()
    axes[1].bar(["Estimated Cost (USD)"], [total_cost], color="#64B5CD")
    axes[1].set_title("Estimated Cost")
    fig.suptitle("Cost / Token Summary")
    fig.tight_layout()
    fig.savefig(plots_dir / "cost_token_summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_result_dirs()
    results_csv = RESULTS_DIR / "evaluation_results.csv"
    if not results_csv.exists():
        logger.error(
            "%s not found. Run `python run_evaluation.py` first.", results_csv
        )
        sys.exit(1)

    df = pd.read_csv(results_csv)
    if "experiment_name" in df.columns and df["experiment_name"].nunique() > 1:
        # Default to the primary experiment for the terminal report if multiple
        # experiments were accidentally combined in one file.
        primary_name = df["experiment_name"].mode().iloc[0]
        df = df[df["experiment_name"] == primary_name]

    summary = summary_from_csv(df)
    save_summary_json(summary, RESULTS_DIR / "evaluation_summary.json")
    print_terminal_report(summary)
    regenerate_plots_from_df(df, PLOTS_DIR)
    logger.info("Report regenerated from %s", results_csv)


if __name__ == "__main__":
    main()
