#!/usr/bin/env python
"""Main entrypoint: run the full RAG evaluation pipeline end to end.

Runs two experiments (top_k=2 and top_k=4), writes per-experiment and
combined CSV/JSON results, generates plots and error analysis, and prints
a terminal report for the primary experiment.

Usage:
    python run_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.chunker import chunk_documents
from src.config import (
    Config,
    DOCUMENTS_DIR,
    EVAL_DATASET_PATH,
    RESULTS_DIR,
    PLOTS_DIR,
    ensure_result_dirs,
)
from src.data_loader import load_documents, load_evaluation_dataset
from src.embeddings import EmbeddingModel
from src.evaluator import EvaluationEngine
from src.generator import AnswerGenerator
from src.logging_config import setup_logging
from src.reporting import (
    compute_summary,
    generate_plots,
    print_terminal_report,
    results_to_dataframe,
    save_error_analysis,
    save_results_csv,
    save_summary_json,
)
from src.retriever import Retriever

logger = setup_logging()


def run_experiment(config: Config, documents, embedding_model: EmbeddingModel):
    """Run one full evaluation experiment and return its results list."""
    chunks = chunk_documents(
        documents,
        chunk_size_words=config.chunk_size_words,
        overlap_words=config.chunk_overlap_words,
    )
    retriever = Retriever(chunks, embedding_model)
    generator = AnswerGenerator(config, embedding_model)
    engine = EvaluationEngine(config, retriever, generator, embedding_model)

    questions = load_evaluation_dataset(EVAL_DATASET_PATH)
    results = engine.evaluate_all(questions)
    return results


def main() -> None:
    ensure_result_dirs()
    logger.info("Starting RAG evaluation pipeline")

    documents = load_documents(DOCUMENTS_DIR)
    embedding_model = EmbeddingModel()

    # Warm the embedding model once so both experiments reuse it.
    embedding_model.encode(["warmup"])

    primary_config = Config(top_k=3, experiment_name="primary_top_k_3")
    primary_results = run_experiment(primary_config, documents, embedding_model)

    exp_a_config = Config(top_k=2, experiment_name="experiment_a_top_k_2")
    exp_a_results = run_experiment(exp_a_config, documents, embedding_model)

    exp_b_config = Config(top_k=4, experiment_name="experiment_b_top_k_4")
    exp_b_results = run_experiment(exp_b_config, documents, embedding_model)

    # --- Primary results: full CSV / summary JSON / plots / error analysis ---
    save_results_csv(primary_results, RESULTS_DIR / "evaluation_results.csv")
    summary = compute_summary(primary_results)
    save_summary_json(summary, RESULTS_DIR / "evaluation_summary.json")
    generate_plots(primary_results, PLOTS_DIR)
    save_error_analysis(primary_results, RESULTS_DIR / "error_analysis.csv")

    # --- Experiment comparison across top_k values ---
    comparison_rows = []
    for name, results in [
        ("experiment_a_top_k_2", exp_a_results),
        ("experiment_b_top_k_4", exp_b_results),
    ]:
        s = compute_summary(results)
        s["experiment_name"] = name
        comparison_rows.append(s)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = RESULTS_DIR / "experiment_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Saved experiment comparison to %s", comparison_path)

    print_terminal_report(summary, title="RAG EVALUATION REPORT (top_k=3)")
    logger.info("Pipeline complete. See the results/ directory for all outputs.")


if __name__ == "__main__":
    main()
