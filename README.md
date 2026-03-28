# CRAG-Ops: Self-Correcting Agentic RAG System

CRAG-Ops is a full-stack Corrective RAG application with a FastAPI backend, a React + Vite frontend, and a LangGraph-powered workflow that evaluates retrieval quality before generation. It supports PDF-grounded question answering and web-grounded general research mode.

## Features

- PDF upload, parsing, chunking, and local ChromaDB indexing
- Retrieval from ChromaDB for uploaded documents or DuckDuckGo for web mode
- Cross-encoder relevance scoring to drive CRAG decisions
- Conditional LangGraph routing for approve, expand, or clarify paths
- OpenAI answer generation with source-aware citations
- Guardrail validation with regeneration on contradiction
- Persistent logs and trace files per request

## Project Structure

```text
backend/
  app/
    evaluator/
    generation/
    graph/
    guardrail/
    ingestion/
    observability/
    retrieval/
frontend/
logs/
storage/
```

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Set `OPENAI_API_KEY` in `backend/.env` before calling the chat endpoint.

## Frontend Setup

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

The frontend expects the API at `http://localhost:8000/api` by default.

## API Endpoints

- `POST /api/upload` accepts a PDF and returns a `document_id`
- `POST /api/chat` accepts:

```json
{
  "query": "What are the main findings?",
  "mode": "pdf",
  "document_id": "optional-document-id"
}
```

Response shape:

```json
{
  "answer": "...",
  "citations": [],
  "trace_id": "...",
  "logs": []
}
```

## Logging

- `logs/app.log`
- `logs/pipeline.log`
- `logs/error.log`
- `logs/traces/{trace_id}.json`

Each pipeline step captures its inputs, outputs, and decision for debugging and observability.
