"""ChromaDB vector store helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb

from app.config import get_settings
from app.retrieval.embeddings import get_embedding_model


class VectorStore:
    """Thin wrapper around Chroma collection operations."""

    def __init__(self) -> None:
        settings = get_settings()
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = client.get_or_create_collection(name="crag_ops_chunks")

    def upsert(self, chunks: list[dict[str, Any]]) -> None:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_embedding_model().encode(texts, convert_to_numpy=True).tolist()
        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in chunks],
            embeddings=embeddings,
        )

    def query(self, query: str, *, top_k: int, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query_embedding = get_embedding_model().encode([query], convert_to_numpy=True).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        matches: list[dict[str, Any]] = []
        for idx, text in enumerate(results.get("documents", [[]])[0]):
            matches.append(
                {
                    "text": text,
                    "metadata": results.get("metadatas", [[]])[0][idx],
                    "distance": results.get("distances", [[]])[0][idx] if results.get("distances") else None,
                }
            )
        return matches


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return a cached vector store instance."""

    return VectorStore()
