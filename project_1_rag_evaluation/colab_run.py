#!/usr/bin/env python
"""Single-command Colab (or local) demo runner.

Performs the complete demo in one call:
  1. Installs/checks dependencies (best effort, skipped if already satisfied).
  2. Loads documents.
  3. Builds embeddings.
  4. Initializes the retriever.
  5. Runs both top_k experiments (2 and 4) plus the primary (top_k=3) run.
  6. Evaluates all test cases.
  7. Generates CSV results.
  8. Generates summary JSON.
  9. Generates plots.
  10. Generates error analysis.
  11. Prints the final report.

Usage:
    python colab_run.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_dependencies() -> None:
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
        import matplotlib  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("Some dependencies are missing; installing from requirements.txt ...")
        requirements_path = PROJECT_ROOT / "requirements.txt"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)],
            check=False,
        )


def main() -> None:
    _ensure_dependencies()

    from src.config import ensure_result_dirs
    from run_evaluation import main as run_evaluation_main

    ensure_result_dirs()
    print("Running the full RAG evaluation pipeline (this may take a minute)...\n")
    run_evaluation_main()
    print("\nDone. Results are available under the results/ directory:")
    print("  - results/evaluation_results.csv")
    print("  - results/evaluation_summary.json")
    print("  - results/experiment_comparison.csv")
    print("  - results/error_analysis.csv")
    print("  - results/plots/*.png")


if __name__ == "__main__":
    main()
