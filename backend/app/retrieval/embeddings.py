"""Embedding and reranker model loading."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the local embedding model lazily."""

    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """Load the local cross-encoder reranker lazily."""

    settings = get_settings()
    return CrossEncoder(settings.reranker_model)
