"""PDF ingestion service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import fitz
from fastapi import UploadFile

from app.config import get_settings
from app.ingestion.chunker import semantic_chunk
from app.retrieval.vector_store import get_vector_store


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    chunk_count: int
    pages: int


async def ingest_pdf(file: UploadFile) -> IngestionResult:
    """Persist, parse, chunk, and index an uploaded PDF."""

    settings = get_settings()
    document_id = uuid4().hex
    suffix = Path(file.filename or "uploaded.pdf").suffix or ".pdf"
    filename = f"{document_id}{suffix}"
    stored_path = settings.upload_path / filename

    content = await file.read()
    stored_path.write_bytes(content)

    document = fitz.open(stream=content, filetype="pdf")
    page_count = document.page_count
    chunks: list[dict] = []

    for page_index in range(page_count):
        page = document.load_page(page_index)
        text = page.get_text("text")
        for chunk_index, chunk in enumerate(semantic_chunk(text)):
            chunks.append(
                {
                    "id": f"{document_id}-{page_index + 1}-{chunk_index}",
                    "text": chunk,
                    "metadata": {
                        "document_id": document_id,
                        "page": page_index + 1,
                        "source": file.filename or "uploaded.pdf",
                    },
                }
            )

    if chunks:
        get_vector_store().upsert(chunks)

    document.close()

    return IngestionResult(
        document_id=document_id,
        filename=file.filename or "uploaded.pdf",
        chunk_count=len(chunks),
        pages=page_count,
    )
