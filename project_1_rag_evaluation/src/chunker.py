"""Simple word-count based chunking with overlap.

The knowledge base documents are intentionally short (a paragraph or two),
so a lightweight sliding-window word chunker is sufficient and keeps the
project free of heavy NLP dependencies.
"""

from __future__ import annotations

from typing import List

from src.logging_config import setup_logging
from src.models import Chunk, Document

logger = setup_logging()


def chunk_document(
    document: Document, chunk_size_words: int = 60, overlap_words: int = 15
) -> List[Chunk]:
    """Split a document's text into overlapping word-window chunks.

    If the document is shorter than chunk_size_words, a single chunk
    containing the whole document is returned.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and < chunk_size_words")

    words = document.text.split()
    if not words:
        return []

    if len(words) <= chunk_size_words:
        return [Chunk(chunk_id=f"{document.doc_id}_c0", doc_id=document.doc_id, text=document.text)]

    chunks: List[Chunk] = []
    step = chunk_size_words - overlap_words
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunk_text = " ".join(window)
        chunks.append(
            Chunk(chunk_id=f"{document.doc_id}_c{index}", doc_id=document.doc_id, text=chunk_text)
        )
        index += 1
        if start + chunk_size_words >= len(words):
            break
        start += step

    return chunks


def chunk_documents(
    documents: List[Document], chunk_size_words: int = 60, overlap_words: int = 15
) -> List[Chunk]:
    """Chunk every document and return a flat list of chunks."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(doc, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
        )
    logger.info("Produced %d chunks from %d documents", len(all_chunks), len(documents))
    return all_chunks
