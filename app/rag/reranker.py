"""Reciprocal Rank Fusion (RRF) hybrid reranker combining FAISS dense similarity and BM25 lexical search."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.rag.chunker import PaperChunk

WORD_RE = re.compile(r"[a-z0-9]+")
BM25_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "which", "when", "where", "who",
    "how", "into", "than", "then", "also", "can", "will", "does", "used", "many",
    "much", "why", "under", "across", "during", "between", "in", "on", "of", "to",
    "is", "by", "at", "be", "or", "as", "an", "do",
}


def tokenize_bm25(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in BM25_STOPWORDS]


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with dense, lexical, and fused confidence scores."""

    chunk: PaperChunk
    dense_score: float
    bm25_score: float
    hybrid_score: float


def fuse_dense_and_bm25(
    query: str,
    chunks: list[PaperChunk],
    dense_scores: list[float],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Combine dense cosine similarities and BM25 lexical scores into a calibrated [0, 1] confidence score."""
    if not chunks:
        return []

    tokenized_corpus = [tokenize_bm25(f"{c.title} {c.section} {c.text}") for c in chunks]
    query_tokens = tokenize_bm25(query)

    if any(tokenized_corpus) and query_tokens:
        bm25 = BM25Okapi(tokenized_corpus)
        raw_bm25 = [max(0.0, float(s)) for s in bm25.get_scores(query_tokens)]
    else:
        raw_bm25 = [0.0] * len(chunks)

    max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
    # Require a minimum floor of 1.5 on max_bm25 so random stopword matches don't normalize to 1.0
    norm_denom = max(max_bm25, 1.5)
    norm_bm25 = [min(1.0, s / norm_denom) if max_bm25 > 1e-6 else 0.0 for s in raw_bm25]

    # Rank orderings for RRF
    dense_ranks = {
        idx: rank for rank, idx in enumerate(sorted(range(len(chunks)), key=lambda i: dense_scores[i], reverse=True))
    }
    bm25_ranks = {
        idx: rank for rank, idx in enumerate(sorted(range(len(chunks)), key=lambda i: norm_bm25[i], reverse=True))
    }

    fused: list[RetrievedChunk] = []
    for idx, chunk in enumerate(chunks):
        d_score = max(0.0, float(dense_scores[idx]))
        b_score = float(norm_bm25[idx])
        has_signal = (d_score > 0.08) or (b_score > 0.12)
        rrf_bonus = (
            (1.0 / (rrf_k + dense_ranks[idx] + 1) + 1.0 / (rrf_k + bm25_ranks[idx] + 1)) * (rrf_k / 2.0)
            if has_signal
            else 0.0
        )
        # Calibrated hybrid confidence combining dense cosine similarity, normalized BM25, and RRF rank agreement
        hybrid = round(min(1.0, 0.55 * d_score + 0.35 * b_score + 0.10 * rrf_bonus), 4)
        fused.append(
            RetrievedChunk(
                chunk=chunk,
                dense_score=round(d_score, 4),
                bm25_score=round(b_score, 4),
                hybrid_score=hybrid,
            )
        )

    fused.sort(key=lambda item: item.hybrid_score, reverse=True)
    return fused[:top_k]
