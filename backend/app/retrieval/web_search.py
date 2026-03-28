"""DuckDuckGo-powered web search."""

from __future__ import annotations

import re

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "who",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _web_relevance(query: str, title: str, body: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    haystack_tokens = _tokenize(f"{title} {body}")
    overlap = len(query_tokens & haystack_tokens)
    return overlap / len(query_tokens)


def search_web(query: str, *, top_k: int = 5) -> list[dict]:
    """Return filtered web results for a query."""

    scored_items: list[tuple[float, dict]] = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=top_k * 4)
        for idx, result in enumerate(results, start=1):
            title = result.get("title", f"Web Result {idx}")
            body = result.get("body", "")
            url = result.get("href")
            score = _web_relevance(query, title, body)
            if score == 0:
                continue
            scored_items.append(
                (
                    score,
                    {
                        "text": f"{title}. {body}".strip(),
                        "metadata": {
                            "source": title,
                            "page": None,
                            "url": url,
                        },
                    },
                )
            )

    scored_items.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored_items[:top_k]]
