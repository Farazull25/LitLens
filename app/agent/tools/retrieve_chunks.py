"""Hybrid FAISS + BM25 chunk retrieval tool with retrieval confidence gating."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config import get_settings
from app.observability.metrics import metrics_registry
from app.rag.vector_store import FAISSVectorStore


class RetrieveChunksInput(BaseModel):
    """Pydantic schema for retrieve_chunks tool input."""

    query: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=15)
    arxiv_ids: list[str] | None = Field(default=None)


def run_retrieve_chunks(
    raw_args: dict[str, Any],
    vector_store: FAISSVectorStore,
) -> dict[str, Any]:
    """Retrieve top-K chunks and compute maximum confidence against the refusal threshold."""
    validated = RetrieveChunksInput.model_validate(raw_args)
    settings = get_settings()

    results = vector_store.search(
        query=validated.query,
        top_k=validated.top_k,
        arxiv_ids=validated.arxiv_ids or None,
    )

    serialized: list[dict[str, Any]] = []
    for item in results:
        c = item.chunk
        serialized.append(
            {
                "chunk_id": c.chunk_id,
                "arxiv_id": c.arxiv_id,
                "title": c.title,
                "authors": c.authors,
                "published": c.published,
                "section": c.section,
                "page_number": c.page_number,
                "text": c.text,
                "dense_score": item.dense_score,
                "bm25_score": item.bm25_score,
                "hybrid_score": item.hybrid_score,
            }
        )

    max_conf = max((r["hybrid_score"] for r in serialized), default=0.0)
    confident = max_conf >= settings.min_retrieval_confidence

    if confident:
        metrics_registry.record("RetrievalHitCount", 1.0)

    return {
        "query": validated.query,
        "chunks": serialized,
        "max_confidence": round(max_conf, 4),
        "threshold": settings.min_retrieval_confidence,
        "is_confident": confident,
    }
