"""Answer generation with OpenAI."""

from __future__ import annotations

from openai import OpenAI

from app.config import get_settings


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a grounded prompt context."""

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {index}")
        page = metadata.get("page")
        url = metadata.get("url")
        label = f"Source {index} | {source}"
        if page:
            label += f" | Page {page}"
        if url:
            label += f" | URL {url}"
        lines.append(f"{label}\n{chunk.get('text', '').strip()}")
    return "\n\n".join(lines)


def _build_openai_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in backend/.env before using /api/chat.")
    return OpenAI(api_key=settings.openai_api_key)


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate a grounded answer with strict citation rules."""

    settings = get_settings()
    client = _build_openai_client()
    context = build_context(chunks)

    prompt = f"""
You are a grounded CRAG answer generator.
Answer the user's question using only the provided context.
If the context is insufficient, say so plainly.
Every factual claim must include citations in the form [Source X, Page Y] or [Source X].
Do not invent sources.

Question:
{query}

Context:
{context}
""".strip()

    response = client.responses.create(
        model=settings.openai_model,
        input=prompt,
        temperature=0.2,
    )
    return response.output_text.strip()
