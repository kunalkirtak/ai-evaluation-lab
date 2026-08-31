#!/usr/bin/env python
"""Google Colab setup helper.

This script is idempotent: it verifies that the expected project
directories exist (creating any that are missing) and installs
requirements.txt if run standalone. It does NOT recreate source files --
those already ship with the project. It exists so a Colab notebook can
call a single `!python colab_setup.py` cell to make sure the environment
is ready before running `colab_run.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_DIRS = [
    PROJECT_ROOT / "data" / "documents",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "results" / "plots",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "tests",
]


def ensure_directories() -> None:
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"OK  {directory}")


def install_requirements() -> None:
    requirements_path = PROJECT_ROOT / "requirements.txt"
    if not requirements_path.exists():
        print("requirements.txt not found; skipping install.")
        return
    print("Installing requirements.txt ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)],
        check=False,
    )


def main() -> None:
    print("Setting up rag-evaluation-benchmark project structure...")
    ensure_directories()
    install_requirements()
    print("Setup complete. Next: run `python run_evaluation.py`.")


if __name__ == "__main__":
    main()
