"""
Benchmark runner: executes the fixed dataset against every configured
provider, evaluates quality, measures performance, calculates cost, and
aggregates everything into per-model and overall results.

Execution is SEQUENTIAL (not concurrent) by design - it keeps the code
simple, reproducible, and easy to reason about for a portfolio project.
"""

import logging
import time
from dataclasses import asdict
from typing import List, Dict, Any, Optional

from src import config, observability
from src.config import PricingConfig, ScoringWeights
from src.dataset import load_dataset
from src.evaluator import SemanticEvaluator, evaluate_answer, optional_llm_judge
from src.metrics import (
    latency_stats,
    calculate_cost,
    success_rate,
    error_rate,
    throughput_rps,
    safe_divide,
    cost_per_successful_answer,
    normalize_min_max,
    composite_production_score,
    pareto_efficient,
)
from src.models import BenchmarkRecord
from src.providers import LLMProvider, LocalMockProvider, build_local_profiles

logger = logging.getLogger("llm_observatory.benchmark")


def _failure_category(record_success: bool, error: Optional[str], is_correct: bool) -> str:
    if record_success and is_correct:
        return "successful"
    if not record_success:
        err = (error or "").lower()
        if "timeout" in err:
            return "timeout"
        if "validation" in err:
            return "validation_error"
        return "api_error"
    # succeeded technically, but quality was below threshold
    return "low_quality"


def run_benchmark(
    providers: Optional[List[LLMProvider]] = None,
    pricing_by_model: Optional[Dict[str, PricingConfig]] = None,
    dataset_path: Optional[str] = None,
    weights: Optional[ScoringWeights] = None,
) -> Dict[str, Any]:
    """
    Run the full benchmark across all providers and return a structured
    results dictionary containing raw records, per-model comparison
    stats, Pareto analysis, and a recommendation.
    """
    config.ensure_result_dirs()

    providers = providers or build_local_profiles()
    weights = weights or ScoringWeights()
    pricing_by_model = pricing_by_model or {p.model_name: PricingConfig() for p in providers}

    dataset = load_dataset(dataset_path)
    evaluator = SemanticEvaluator()

    logger.info("Benchmark started")
    logger.info(f"Providers={[p.model_name for p in providers]} | test_cases={len(dataset)}")

    records: List[BenchmarkRecord] = []
    run_start = time.perf_counter()

    judge_fn = optional_llm_judge if config.ENABLE_LLM_JUDGE else None

    for provider in providers:
        pricing = pricing_by_model.get(provider.model_name, PricingConfig())
        for case in dataset:
            with observability.trace_run(f"{provider.model_name}:{case.id}"):
                if isinstance(provider, LocalMockProvider):
                    gen = provider.generate(case.question, reference_hint=case.reference_answer)
                else:
                    gen = provider.generate(case.question)

                logger.info(
                    f"Model={provider.model_name} | request={case.id} | "
                    f"success={gen.success} | latency={gen.latency_seconds:.4f}"
                )

                eval_result = evaluate_answer(
                    question=case.question,
                    generated_answer=gen.answer,
                    reference_answer=case.reference_answer,
                    evaluator=evaluator,
                    judge_fn=judge_fn,
                )

                cost = calculate_cost(gen.input_tokens, gen.output_tokens, pricing)

                effective_correct = gen.success and eval_result.is_correct
                category = _failure_category(gen.success, gen.error, eval_result.is_correct)

                record = BenchmarkRecord(
                    model_name=provider.model_name,
                    question_id=case.id,
                    question=case.question,
                    category=case.category,
                    reference_answer=case.reference_answer,
                    generated_answer=gen.answer,
                    input_tokens=gen.input_tokens,
                    output_tokens=gen.output_tokens,
                    total_tokens=gen.total_tokens,
                    token_source=gen.token_source,
                    latency_seconds=gen.latency_seconds,
                    success=gen.success,
                    error=gen.error,
                    quality_score=eval_result.quality_score if gen.success else 0.0,
                    relevance_score=eval_result.relevance_score if gen.success else 0.0,
                    is_correct=effective_correct,
                    input_cost=cost["input_cost"],
                    output_cost=cost["output_cost"],
                    total_cost=cost["total_cost"],
                    failure_category=category,
                )
                records.append(record)

    run_duration = time.perf_counter() - run_start
    logger.info(f"Benchmark finished | duration={run_duration:.3f}s | total_requests={len(records)}")

    comparison = aggregate_by_model(records, run_duration)
    pareto_names = pareto_efficient(
        [
            {
                "name": row["model"],
                "quality": row["mean_quality"],
                "mean_latency": row["mean_latency"],
                "cost_per_request": row["cost_per_request"],
            }
            for row in comparison
        ]
    )

    comparison = attach_production_scores(comparison, weights)
    recommendation = build_recommendation(comparison, pareto_names)

    return {
        "records": records,
        "comparison": comparison,
        "pareto_efficient": pareto_names,
        "recommendation": recommendation,
        "run_duration_seconds": run_duration,
        "test_case_count": len(dataset),
        "provider_count": len(providers),
    }


def aggregate_by_model(records: List[BenchmarkRecord], run_duration: float) -> List[Dict[str, Any]]:
    """Aggregate raw per-request records into per-model comparison rows."""
    by_model: Dict[str, List[BenchmarkRecord]] = {}
    for r in records:
        by_model.setdefault(r.model_name, []).append(r)

    rows = []
    for model_name, recs in by_model.items():
        total = len(recs)
        successes = [r for r in recs if r.success]
        failures = [r for r in recs if not r.success]
        latencies = [r.latency_seconds for r in recs]
        lat_stats = latency_stats(latencies)

        quality_scores = [r.quality_score for r in successes] or [0.0]
        relevance_scores = [r.relevance_score for r in successes] or [0.0]

        total_cost = sum(r.total_cost for r in recs)
        n_success = len(successes)

        row = {
            "model": model_name,
            "requests": total,
            "success_rate": round(success_rate(n_success, total), 4),
            "error_rate": round(error_rate(len(failures), total), 4),
            "mean_quality": round(sum(quality_scores) / len(quality_scores), 4),
            "mean_relevance": round(sum(relevance_scores) / len(relevance_scores), 4),
            "mean_latency": round(lat_stats["mean"], 5),
            "median_latency": round(lat_stats["median"], 5),
            "p95_latency": round(lat_stats["p95"], 5),
            "average_input_tokens": round(sum(r.input_tokens for r in recs) / total, 2),
            "average_output_tokens": round(sum(r.output_tokens for r in recs) / total, 2),
            "average_total_tokens": round(sum(r.total_tokens for r in recs) / total, 2),
            "total_cost": round(total_cost, 6),
            "cost_per_request": round(safe_divide(total_cost, total), 6),
            "cost_per_successful_answer": round(cost_per_successful_answer(total_cost, n_success), 6),
            "throughput": round(throughput_rps(total, run_duration), 4),
        }
        rows.append(row)

    rows.sort(key=lambda r: r["model"])
    return rows


def attach_production_scores(comparison: List[Dict[str, Any]], weights: ScoringWeights) -> List[Dict[str, Any]]:
    """Compute the composite production score for each model row."""
    if not comparison:
        return comparison

    latencies = [row["mean_latency"] for row in comparison]
    costs = [row["cost_per_request"] for row in comparison]
    lat_min, lat_max = min(latencies), max(latencies)
    cost_min, cost_max = min(costs), max(costs)

    weight_dict = weights.as_dict()

    for row in comparison:
        normalized_speed = normalize_min_max(row["mean_latency"], lat_min, lat_max, invert=True)
        normalized_cost_efficiency = normalize_min_max(row["cost_per_request"], cost_min, cost_max, invert=True)
        row["production_score"] = round(
            composite_production_score(
                quality=row["mean_quality"],
                reliability=row["success_rate"],
                normalized_speed=normalized_speed,
                normalized_cost_efficiency=normalized_cost_efficiency,
                weights=weight_dict,
            ),
            4,
        )
    return comparison


def build_recommendation(comparison: List[Dict[str, Any]], pareto_names: List[str]) -> Dict[str, Any]:
    """Derive a simple, measurement-based recommendation summary."""
    if not comparison:
        return {}

    best_quality = max(comparison, key=lambda r: r["mean_quality"])
    fastest = min(comparison, key=lambda r: r["mean_latency"])
    cheapest = min(comparison, key=lambda r: r["cost_per_request"])
    most_reliable = max(comparison, key=lambda r: r["success_rate"])
    best_balanced = max(comparison, key=lambda r: r["production_score"])

    return {
        "best_quality": best_quality["model"],
        "fastest": fastest["model"],
        "cheapest": cheapest["model"],
        "most_reliable": most_reliable["model"],
        "best_balanced": best_balanced["model"],
        "pareto_efficient": pareto_names,
    }
