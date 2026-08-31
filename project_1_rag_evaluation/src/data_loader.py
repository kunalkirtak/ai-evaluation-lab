"""Loading of source documents and the evaluation dataset from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.logging_config import setup_logging
from src.models import Document, EvalQuestion

logger = setup_logging()


def load_documents(documents_dir: Path) -> List[Document]:
    """Load every .txt file in documents_dir as a Document.

    The doc_id is derived from the filename stem (e.g. doc_001.txt -> doc_001).
    Files are sorted so that loading is deterministic.
    """
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    paths = sorted(documents_dir.glob("*.txt"))
    if not paths:
        raise ValueError(f"No .txt documents found in {documents_dir}")

    documents: List[Document] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error("Failed to read document %s: %s", path, exc)
            continue
        if not text:
            logger.warning("Skipping empty document: %s", path)
            continue
        documents.append(Document(doc_id=path.stem, text=text, source_path=str(path)))

    logger.info("Loaded %d documents from %s", len(documents), documents_dir)
    return documents


def load_evaluation_dataset(dataset_path: Path) -> List[EvalQuestion]:
    """Load the evaluation dataset JSON file into EvalQuestion objects."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    questions: List[EvalQuestion] = []
    for item in raw:
        try:
            questions.append(
                EvalQuestion(
                    id=item["id"],
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                    relevant_document_ids=list(item["relevant_document_ids"]),
                )
            )
        except KeyError as exc:
            logger.error("Malformed evaluation example %s: missing %s", item, exc)
            continue

    logger.info("Loaded %d evaluation questions from %s", len(questions), dataset_path)
    return questions
