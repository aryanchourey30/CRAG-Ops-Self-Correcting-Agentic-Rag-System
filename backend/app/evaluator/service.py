"""Retrieval quality evaluation."""

from __future__ import annotations

import math
import re
from statistics import mean

from app.retrieval.embeddings import get_reranker_model

APPROVE_THRESHOLD = 0.45
EXPAND_THRESHOLD = 0.2
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "pdf",
    "tell",
    "the",
    "this",
    "to",
    "what",
    "which",
    "who",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _lexical_overlap_score(query: str, chunk_text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    chunk_tokens = _tokenize(chunk_text)
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


def _distance_score(chunk: dict) -> float:
    distance = chunk.get("distance")
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1 / (1 + float(distance))))


def evaluate_retrieval(query: str, chunks: list[dict]) -> tuple[float, str]:
    """Score retrieved chunks and return CRAG decision."""

    if not chunks:
        return 0.0, "REJECT"

    usable_chunks = [chunk for chunk in chunks if chunk.get("text", "").strip()]
    if not usable_chunks:
        return 0.0, "REJECT"

    pairs = [[query, chunk["text"]] for chunk in usable_chunks]
    raw_scores = get_reranker_model().predict(pairs)
    reranker_scores = [1 / (1 + math.exp(-float(score))) for score in raw_scores]
    lexical_scores = [_lexical_overlap_score(query, chunk["text"]) for chunk in usable_chunks]
    distance_scores = [_distance_score(chunk) for chunk in usable_chunks]

    average_reranker = mean(reranker_scores)
    best_reranker = max(reranker_scores)
    best_lexical = max(lexical_scores)
    best_distance = max(distance_scores)

    blended_score = round(
        (average_reranker * 0.2)
        + (best_reranker * 0.15)
        + (best_lexical * 0.45)
        + (best_distance * 0.2),
        4,
    )

    if best_lexical >= 0.34:
        blended_score = max(blended_score, 0.55)

    if blended_score >= APPROVE_THRESHOLD:
        return blended_score, "APPROVE"
    if blended_score >= EXPAND_THRESHOLD:
        return blended_score, "EXPAND"
    return blended_score, "REJECT"
