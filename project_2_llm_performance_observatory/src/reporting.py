"""
Generates all benchmark output artifacts:

  results/benchmark_results.csv   - every raw request record
  results/model_comparison.csv    - aggregated per-model comparison
  results/error_analysis.csv      - failed / low-quality requests
  results/summary.json            - overall summary + recommendation
  results/run_config.json         - reproducibility metadata
  results/plots/*.png             - comparison charts
  terminal report                 - concise human-readable summary
"""

import csv
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from src import config

logger = logging.getLogger("llm_observatory.reporting")

RECORD_FIELDS = [
    "model_name", "question_id", "question", "category", "reference_answer",
    "generated_answer", "input_tokens", "output_tokens", "total_tokens",
    "token_source", "latency_seconds", "success", "error", "quality_score",
    "relevance_score", "is_correct", "input_cost", "output_cost", "total_cost",
    "failure_category",
]

COMPARISON_FIELDS = [
    "model", "requests", "success_rate", "error_rate", "mean_quality",
    "mean_relevance", "mean_latency", "median_latency", "p95_latency",
    "average_input_tokens", "average_output_tokens", "average_total_tokens",
    "total_cost", "cost_per_request", "cost_per_successful_answer",
    "throughput", "production_score",
]

ERROR_FIELDS = [
    "model", "question_id", "question", "generated_answer", "reference_answer",
    "quality_score", "latency", "error", "failure_category",
]


def write_benchmark_results_csv(records: List[Any], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for r in records:
            row = asdict(r)
            writer.writerow({k: row[k] for k in RECORD_FIELDS})
    logger.info(f"Wrote {path}")


def write_model_comparison_csv(comparison: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in comparison:
            writer.writerow({k: row.get(k, "") for k in COMPARISON_FIELDS})
    logger.info(f"Wrote {path}")


def write_error_analysis_csv(records: List[Any], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        for r in records:
            if r.failure_category == "successful":
                continue
            writer.writerow({
                "model": r.model_name,
                "question_id": r.question_id,
                "question": r.question,
                "generated_answer": r.generated_answer,
                "reference_answer": r.reference_answer,
                "quality_score": r.quality_score,
                "latency": r.latency_seconds,
                "error": r.error or "",
                "failure_category": r.failure_category,
            })
    logger.info(f"Wrote {path}")


def write_summary_json(result: Dict[str, Any], path: str) -> None:
    summary = {
        "test_case_count": result["test_case_count"],
        "provider_count": result["provider_count"],
        "run_duration_seconds": round(result["run_duration_seconds"], 4),
        "comparison": result["comparison"],
        "pareto_efficient": result["pareto_efficient"],
        "recommendation": result["recommendation"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {path}")


def write_run_config_json(
    path: str,
    mode: str,
    model_configs: List[str],
    weights: Dict[str, float],
    pricing_note: str = "Pricing values are example/configurable benchmark assumptions.",
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model_configurations": model_configs,
        "quality_threshold": config.QUALITY_THRESHOLD,
        "relevance_threshold": config.RELEVANCE_THRESHOLD,
        "scoring_weights": weights,
        "pricing_assumption_note": pricing_note,
        "default_input_cost_per_1m_tokens": config.DEFAULT_INPUT_COST_PER_1M,
        "default_output_cost_per_1m_tokens": config.DEFAULT_OUTPUT_COST_PER_1M,
        "random_seed": config.RANDOM_SEED,
        "dataset_path": config.DATA_PATH,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote {path}")


def _save_bar_chart(labels, values, title, ylabel, path, color=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=color or "#4C72B0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model configuration")
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.4g}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info(f"Wrote {path}")


def generate_plots(comparison: List[Dict[str, Any]], plots_dir: str) -> None:
    try:
        models = [row["model"] for row in comparison]

        _save_bar_chart(
            models, [row["mean_quality"] for row in comparison],
            "Mean Quality by Configuration", "Quality score (0-1)",
            os.path.join(plots_dir, "quality_comparison.png"), color="#55A868",
        )
        _save_bar_chart(
            models, [row["mean_latency"] for row in comparison],
            "Mean Latency by Configuration", "Latency (seconds)",
            os.path.join(plots_dir, "latency_comparison.png"), color="#C44E52",
        )
        _save_bar_chart(
            models, [row["cost_per_request"] for row in comparison],
            "Cost per Request by Configuration", "Cost (USD)",
            os.path.join(plots_dir, "cost_comparison.png"), color="#8172B2",
        )
        _save_bar_chart(
            models, [row["average_total_tokens"] for row in comparison],
            "Average Total Tokens by Configuration", "Tokens",
            os.path.join(plots_dir, "token_comparison.png"), color="#CCB974",
        )
        generate_quality_vs_cost_plot(comparison, os.path.join(plots_dir, "quality_vs_cost.png"))
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Plot generation failed (continuing without plots): {exc}")


def generate_quality_vs_cost_plot(comparison: List[Dict[str, Any]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in comparison:
        ax.scatter(row["cost_per_request"], row["mean_quality"], s=120, alpha=0.8)
        ax.annotate(
            row["model"],
            (row["cost_per_request"], row["mean_quality"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    ax.set_xlabel("Cost per Request (USD)")
    ax.set_ylabel("Quality Score (0-1)")
    ax.set_title("Quality vs. Cost Tradeoff")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info(f"Wrote {path}")


def print_terminal_report(result: Dict[str, Any], mode: str) -> None:
    comparison = result["comparison"]
    rec = result["recommendation"]
    bar = "=" * 60

    print(bar)
    print("LLM PERFORMANCE OBSERVATORY")
    print("=" * 27)
    print()
    print(f"Benchmark Mode: {mode}")
    print()
    print(f"Configurations: {result['provider_count']}")
    print(f"Test Cases: {result['test_case_count']}")
    print()

    if not comparison:
        print("No results generated.")
        print(bar)
        return

    best_quality_row = max(comparison, key=lambda r: r["mean_quality"])
    fastest_row = min(comparison, key=lambda r: r["mean_latency"])
    cheapest_row = min(comparison, key=lambda r: r["cost_per_request"])
    most_reliable_row = max(comparison, key=lambda r: r["success_rate"])

    print("QUALITY")
    print(f"Best Quality: {best_quality_row['model']} ({best_quality_row['mean_quality']:.3f})")
    print()
    print("PERFORMANCE")
    print(f"Lowest Mean Latency: {fastest_row['model']} ({fastest_row['mean_latency']:.4f}s)")
    print()
    print("ECONOMICS")
    print(f"Lowest Cost: {cheapest_row['model']} (${cheapest_row['cost_per_request']:.6f}/request)")
    print()
    print("RELIABILITY")
    print(f"Best Success Rate: {most_reliable_row['model']} ({most_reliable_row['success_rate']:.1%})")
    print()
    print("PARETO-EFFICIENT CONFIGURATIONS")
    print(", ".join(result["pareto_efficient"]) or "none")
    print()
    print("RECOMMENDATION")
    print(f"Best balanced (composite production score): {rec.get('best_balanced', 'n/a')}")
    print(
        "Note: 'best' depends on your workload's priorities. Review "
        "results/model_comparison.csv and results/plots/quality_vs_cost.png "
        "before deciding."
    )
    print()
    print(bar)


def generate_all_reports(result: Dict[str, Any], mode: str, weights: Dict[str, float]) -> None:
    config.ensure_result_dirs()

    write_benchmark_results_csv(result["records"], os.path.join(config.RESULTS_DIR, "benchmark_results.csv"))
    write_model_comparison_csv(result["comparison"], os.path.join(config.RESULTS_DIR, "model_comparison.csv"))
    write_error_analysis_csv(result["records"], os.path.join(config.RESULTS_DIR, "error_analysis.csv"))
    write_summary_json(result, os.path.join(config.RESULTS_DIR, "summary.json"))
    write_run_config_json(
        os.path.join(config.RESULTS_DIR, "run_config.json"),
        mode=mode,
        model_configs=[row["model"] for row in result["comparison"]],
        weights=weights,
    )
    generate_plots(result["comparison"], config.PLOTS_DIR)
    print_terminal_report(result, mode)
