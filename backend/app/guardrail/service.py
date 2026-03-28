"""Grounding guardrail checks."""

from __future__ import annotations

from openai import OpenAI

from app.config import get_settings
from app.generation.service import build_context, generate_answer


def _build_openai_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in backend/.env before using /api/chat.")
    return OpenAI(api_key=settings.openai_api_key)


def validate_answer(query: str, answer: str, chunks: list[dict]) -> tuple[str, bool]:
    """Validate whether the answer contradicts context and regenerate once if needed."""

    settings = get_settings()
    client = _build_openai_client()
    context = build_context(chunks)
    prompt = f"""
You are a factuality validator.
Return only VALID or INVALID.
Mark INVALID if the answer contains unsupported claims or contradicts the context.

Question:
{query}

Context:
{context}

Answer:
{answer}
""".strip()

    verdict = client.responses.create(
        model=settings.openai_model,
        input=prompt,
        temperature=0,
    ).output_text.strip().upper()

    if "INVALID" in verdict:
        return generate_answer(query, chunks), True
    return answer, False
