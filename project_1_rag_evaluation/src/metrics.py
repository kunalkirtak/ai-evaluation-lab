"""Deterministic evaluation metrics for retrieval and generation quality.

Retrieval metrics (Precision@K, Recall@K, MRR, Hit Rate@K) are exact,
rule-based computations over document-ID overlap and require no models.

Generation metrics (answer accuracy, answer relevance, faithfulness) are
model-based: they rely on sentence-embedding cosine similarity as a proxy
for semantic agreement. This is a deliberate, documented design choice --
see README.md, section "Deterministic Metrics vs Model-Based Evaluation".
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from src.embeddings import EmbeddingModel, cosine_similarity
from src.generator import split_sentences
from src.models import GenerationMetrics, RetrievalMetrics
from src.config import Config


# ----------------------------------------------------------------------
# Retrieval metrics
# ----------------------------------------------------------------------
def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of retrieved documents that are relevant.

    precision@k = |relevant retrieved| / |retrieved|
    """
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of all relevant documents that were retrieved.

    recall@k = |relevant retrieved| / |total relevant|
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant document in the retrieved list.

    Returns 0.0 if no relevant document was retrieved.
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """1.0 if at least one relevant document was retrieved, else 0.0."""
    relevant_set = set(relevant_ids)
    return 1.0 if any(doc_id in relevant_set for doc_id in retrieved_ids) else 0.0


def compute_retrieval_metrics(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str]
) -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids),
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
        hit_rate_at_k=hit_rate_at_k(retrieved_ids, relevant_ids),
    )


# ----------------------------------------------------------------------
# Generation metrics (embedding-based proxies)
# ----------------------------------------------------------------------
def semantic_similarity(text_a: str, text_b: str, embedding_model: EmbeddingModel) -> float:
    """Cosine similarity between the sentence embeddings of two texts.

    Returns a value clipped to [0, 1] (cosine similarity of normalized
    sentence-transformer embeddings is typically already in this range for
    semantically related text, but we clip defensively).
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vecs = embedding_model.encode([text_a, text_b])
    score = cosine_similarity(vecs[0], vecs[1])
    return max(0.0, min(1.0, score))


def answer_accuracy(generated_answer: str, ground_truth: str, embedding_model: EmbeddingModel) -> float:
    """Semantic similarity between the generated answer and the ground truth."""
    return semantic_similarity(generated_answer, ground_truth, embedding_model)


def answer_relevance(question: str, generated_answer: str, embedding_model: EmbeddingModel) -> float:
    """Semantic similarity between the question and the generated answer."""
    return semantic_similarity(question, generated_answer, embedding_model)


def faithfulness(
    generated_answer: str,
    context_chunks: List[str],
    embedding_model: EmbeddingModel,
    support_threshold: float = 0.5,
) -> Tuple[float, int, int]:
    """Estimate how much of the answer is supported by the retrieved context.

    Implementation:
      1. Split the answer into sentences.
      2. Embed each answer sentence and each context chunk.
      3. For each answer sentence, take its MAXIMUM similarity to any
         context chunk.
      4. A sentence is "supported" if that max similarity >= support_threshold.
      5. faithfulness_score = supported_sentences / total_sentences.

    Returns (faithfulness_score, unsupported_sentence_count, total_sentence_count).

    This is a project-level proxy for faithfulness, not a formal
    entailment or fact-checking system. See README for limitations.
    """
    sentences = split_sentences(generated_answer)
    if not sentences:
        return 0.0, 0, 0
    if not context_chunks:
        return 0.0, len(sentences), len(sentences)

    sentence_vecs = embedding_model.encode(sentences)
    context_vecs = embedding_model.encode(context_chunks)

    supported = 0
    for i in range(len(sentences)):
        max_sim = max(
            cosine_similarity(sentence_vecs[i], context_vecs[j])
            for j in range(len(context_chunks))
        )
        if max_sim >= support_threshold:
            supported += 1

    total = len(sentences)
    unsupported = total - supported
    score = supported / total
    return score, unsupported, total


def hallucination_rate(faithfulness_score: float) -> float:
    """hallucination_rate = 1 - faithfulness_score (a project-level proxy)."""
    return 1.0 - faithfulness_score


def compute_generation_metrics(
    question: str,
    generated_answer: str,
    ground_truth: str,
    context_chunks: List[str],
    embedding_model: EmbeddingModel,
    config: Config,
) -> GenerationMetrics:
    accuracy = answer_accuracy(generated_answer, ground_truth, embedding_model)
    relevance = answer_relevance(question, generated_answer, embedding_model)
    faith_score, unsupported, total = faithfulness(
        generated_answer,
        context_chunks,
        embedding_model,
        support_threshold=config.faithfulness_support_threshold,
    )
    return GenerationMetrics(
        answer_accuracy=accuracy,
        answer_relevance=relevance,
        faithfulness_score=faith_score,
        hallucination_rate=hallucination_rate(faith_score),
        unsupported_sentence_count=unsupported,
        total_sentence_count=total,
    )
