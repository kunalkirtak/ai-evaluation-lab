"""
Loads the fixed benchmark dataset used by every experiment run.

The dataset is small (~18 questions) and intentionally scoped to a single
coherent technical domain (LLMs / RAG / embeddings / vector databases /
AI agents) so that quality evaluation via semantic similarity is
meaningful.
"""

import json
import os
from typing import List

from src.models import TestCase
from src import config


def load_dataset(path: str = None) -> List[TestCase]:
    path = path or config.DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cases = [
        TestCase(
            id=item["id"],
            question=item["question"],
            reference_answer=item["reference_answer"],
            category=item["category"],
        )
        for item in raw
    ]
    return cases


if __name__ == "__main__":
    data = load_dataset()
    print(f"Loaded {len(data)} test cases")
    for c in data[:3]:
        print(c)
