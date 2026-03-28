"""Retrieval orchestration."""

from __future__ import annotations

from app.config import get_settings
from app.retrieval.vector_store import get_vector_store
from app.retrieval.web_search import search_web


def retrieve_chunks(query: str, *, document_id: str | None = None) -> list[dict]:
    """Retrieve relevant chunks from Chroma or the open web."""

    settings = get_settings()
    if document_id:
        return get_vector_store().query(
            query,
            top_k=settings.retrieval_top_k,
            where={"document_id": document_id},
        )
    return search_web(query, top_k=settings.retrieval_top_k)
