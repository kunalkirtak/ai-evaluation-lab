"""Summary statistics, terminal reporting, plots, and error analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.logging_config import setup_logging
from src.models import EvaluationResult

logger = setup_logging()

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


def results_to_dataframe(results: List[EvaluationResult]) -> pd.DataFrame:
    """Convert a list of EvaluationResult into a flat pandas DataFrame."""
    rows = [r.to_flat_dict() for r in results]
    return pd.DataFrame(rows)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def compute_summary(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Compute the aggregate summary dict described in the project spec."""
    if not results:
        return {"num_test_cases": 0}

    df = results_to_dataframe(results)
    latencies = df["total_latency_s"].tolist()

    summary = {
        "num_test_cases": len(results),
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
    return summary


def print_terminal_report(summary: Dict[str, Any], title: str = "RAG EVALUATION REPORT") -> None:
    """Print a clean, human-readable terminal report."""
    line = "=" * 60
    print(line)
    print(title)
    print("=" * len(title))
    print()
    print(f"Test Cases          : {summary.get('num_test_cases', 0)}")
    print()
    print("Retrieval")
    print(f"Precision@K         : {summary.get('mean_precision_at_k', 0):.2f}")
    print(f"Recall@K            : {summary.get('mean_recall_at_k', 0):.2f}")
    print(f"MRR                 : {summary.get('mean_mrr', 0):.2f}")
    print(f"Hit Rate@K          : {summary.get('mean_hit_rate_at_k', 0):.2f}")
    print()
    print("Generation")
    print(f"Answer Accuracy     : {summary.get('mean_answer_accuracy', 0):.2f}")
    print(f"Faithfulness        : {summary.get('mean_faithfulness', 0):.2f}")
    print(f"Answer Relevance    : {summary.get('mean_answer_relevance', 0):.2f}")
    print(f"Hallucination Rate  : {summary.get('mean_hallucination_rate', 0):.2f}")
    print()
    print("Performance")
    print(f"Mean Latency        : {summary.get('mean_latency_seconds', 0):.4f}s")
    print(f"Median Latency      : {summary.get('median_latency_seconds', 0):.4f}s")
    print(f"P95 Latency         : {summary.get('p95_latency_seconds', 0):.4f}s")
    print()
    print("Cost")
    print(f"Estimated Tokens    : {summary.get('total_estimated_tokens', 0)}")
    print(f"Estimated Cost      : ${summary.get('estimated_total_cost_usd', 0):.6f}")
    print()
    print(line)


def save_results_csv(results: List[EvaluationResult], path: Path) -> None:
    df = results_to_dataframe(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved evaluation results CSV to %s", path)


def save_summary_json(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved evaluation summary JSON to %s", path)


def build_error_analysis(results: List[EvaluationResult]) -> pd.DataFrame:
    """Build the error_analysis.csv content for every example.

    Every example is included with its failure_category so that
    'correct_answer' examples are visible alongside failures for context.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "question_id": r.question_id,
                "question": r.question,
                "expected_answer": r.ground_truth,
                "generated_answer": r.generated_answer,
                "retrieved_doc_ids": ";".join(x.doc_id for x in r.retrieved),
                "top_retrieval_score": r.retrieved[0].score if r.retrieved else 0.0,
                "accuracy_score": r.generation_metrics.answer_accuracy,
                "faithfulness_score": r.generation_metrics.faithfulness_score,
                "relevance_score": r.generation_metrics.answer_relevance,
                "failure_category": r.failure_category,
            }
        )
    return pd.DataFrame(rows)


def save_error_analysis(results: List[EvaluationResult], path: Path) -> None:
    df = build_error_analysis(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved error analysis CSV to %s", path)


# ----------------------------------------------------------------------
# Visualizations
# ----------------------------------------------------------------------
def generate_plots(results: List[EvaluationResult], plots_dir: Path) -> None:
    """Generate the four required plots into plots_dir. No-op if matplotlib missing."""
    if not _HAS_MPL:
        logger.warning("matplotlib not available; skipping plot generation.")
        return
    if not results:
        logger.warning("No results to plot.")
        return

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(results)

    _plot_retrieval_metrics(df, plots_dir / "retrieval_metrics.png")
    _plot_generation_metrics(df, plots_dir / "generation_metrics.png")
    _plot_latency_distribution(df, plots_dir / "latency_distribution.png")
    _plot_cost_token_summary(df, plots_dir / "cost_token_summary.png")


def _plot_retrieval_metrics(df: pd.DataFrame, path: Path) -> None:
    metrics = ["precision_at_k", "recall_at_k", "mrr", "hit_rate_at_k"]
    labels = ["Precision@K", "Recall@K", "MRR", "Hit Rate@K"]
    means = [df[m].mean() for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Metrics Comparison")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_generation_metrics(df: pd.DataFrame, path: Path) -> None:
    metrics = ["answer_accuracy", "faithfulness_score", "answer_relevance", "hallucination_rate"]
    labels = ["Answer Accuracy", "Faithfulness", "Answer Relevance", "Hallucination Rate"]
    means = [df[m].mean() for m in metrics]
    colors = ["#55A868", "#55A868", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Generation Quality Metrics")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_latency_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["total_latency_s"], bins=min(10, max(3, len(df) // 2)), color="#8172B2", edgecolor="white")
    ax.set_xlabel("Total Latency (seconds)")
    ax.set_ylabel("Number of Questions")
    ax.set_title("Latency Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cost_token_summary(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    token_totals = [df["input_tokens"].sum(), df["output_tokens"].sum()]
    axes[0].bar(["Input Tokens", "Output Tokens"], token_totals, color="#CCB974")
    axes[0].set_title("Estimated Token Usage")
    axes[0].set_ylabel("Tokens")

    total_cost = df["estimated_cost_usd"].sum()
    axes[1].bar(["Estimated Cost (USD)"], [total_cost], color="#64B5CD")
    axes[1].set_title("Estimated Cost")
    axes[1].set_ylabel("USD")

    fig.suptitle("Cost / Token Summary")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
