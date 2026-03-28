"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AuthenticationError

from app.config import get_settings
from app.graph.pipeline import run_pipeline
from app.ingestion.pdf_service import ingest_pdf
from app.observability.logger import configure_logging
from app.schemas import ChatRequest, ChatResponse, UploadResponse

settings = get_settings()
configure_logging(settings.log_path)

app_logger = logging.getLogger("crag_ops")
error_logger = logging.getLogger("crag_ops.error")

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Healthcheck endpoint."""

    return {"status": "ok"}


@app.post(f"{settings.api_prefix}/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and ingest a PDF file."""

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    try:
        result = await ingest_pdf(file)
        app_logger.info("upload_pdf | document_id=%s | filename=%s", result.document_id, result.filename)
        return UploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
            pages=result.pages,
        )
    except Exception as exc:  # pragma: no cover
        error_logger.exception("upload_pdf failed: %s", exc)
        raise HTTPException(status_code=500, detail="PDF ingestion failed.") from exc


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the CRAG workflow for a user query."""

    try:
        result = run_pipeline(request.query, mode=request.mode, document_id=request.document_id)
        app_logger.info("chat | trace_id=%s | mode=%s", result["trace_id"], request.mode)
        return ChatResponse(
            answer=result.get("generated_answer", ""),
            citations=result.get("citations", []),
            trace_id=result["trace_id"],
            logs=result.get("logs", []),
        )
    except (RuntimeError, AuthenticationError) as exc:
        error_logger.exception("chat configuration failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        error_logger.exception("chat failed: %s", exc)
        raise HTTPException(status_code=500, detail="CRAG pipeline failed.") from exc
