"""LangGraph state definition."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class GraphState(TypedDict, total=False):
    """State carried across the CRAG workflow."""

    query: str
    mode: Literal["pdf", "web"]
    document_id: str | None
    retrieved_chunks: list[dict[str, Any]]
    relevance_score: float
    decision: str
    generated_answer: str
    citations: list[dict[str, Any]]
    trace_id: str
    logs: list[dict[str, Any]]
    web_search_attempted: bool
