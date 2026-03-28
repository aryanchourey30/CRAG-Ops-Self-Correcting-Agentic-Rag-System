"""Pydantic request and response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload for question answering."""

    query: str = Field(min_length=3)
    mode: Literal["pdf", "web"] = "web"
    document_id: str | None = None


class Citation(BaseModel):
    """Citation entry returned to the client."""

    source: str
    page: int | None = None
    snippet: str | None = None
    url: str | None = None


class ChatResponse(BaseModel):
    """Structured answer returned by the CRAG pipeline."""

    answer: str
    citations: list[Citation]
    trace_id: str
    logs: list[dict[str, Any]]


class UploadResponse(BaseModel):
    """Upload response for PDF ingestion."""

    document_id: str
    filename: str
    chunk_count: int
    pages: int
