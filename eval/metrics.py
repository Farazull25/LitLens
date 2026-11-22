"""Retrieval hit-rate, citation accuracy, and faithfulness evaluation metrics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

WORD_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{3,}")
STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "which", "when", "where", "who",
    "how", "into", "than", "then", "also", "can", "will", "based", "paper", "using",
}


@dataclass(frozen=True)
class SampleEvalScore:
    """Evaluation scores for a single golden Q&A item."""

    question_id: str
    retrieval_hit: float
    reciprocal_rank: float
    citation_precision: float
    citation_recall: float
    faithfulness: float
    correct_refusal: bool
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_retrieval_hit_and_mrr(
    retrieved_arxiv_ids: list[str],
    expected_arxiv_id: str | None,
) -> tuple[float, float]:
    """Compute HitRate@K (0.0 or 1.0) and Reciprocal Rank (1/rank)."""
    if not expected_arxiv_id:
        return (1.0 if not retrieved_arxiv_ids else 0.0), 1.0

    for idx, aid in enumerate(retrieved_arxiv_ids, start=1):
        if aid == expected_arxiv_id:
            return 1.0, round(1.0 / idx, 4)
    return 0.0, 0.0


def compute_citation_accuracy(
    cited_chunk_ids: list[str],
    retrieved_chunk_ids: set[str],
    expected_arxiv_id: str | None,
) -> tuple[float, float]:
    """Compute (citation_precision, citation_recall).

    - Precision: fraction of cited chunk IDs that both exist in retrieved_chunk_ids and match expected_arxiv_id.
    - Recall: 1.0 if expected_arxiv_id is cited (or if expected_arxiv_id is None and no citations are made).
    """
    if expected_arxiv_id is None:
        score = 1.0 if len(cited_chunk_ids) == 0 else 0.0
        return score, score

    if not cited_chunk_ids:
        return 0.0, 0.0

    valid_hits = [
        cid
        for cid in cited_chunk_ids
        if cid in retrieved_chunk_ids and cid.startswith(f"{expected_arxiv_id}#")
    ]
    precision = round(len(valid_hits) / len(cited_chunk_ids), 4)
    recall = 1.0 if len(valid_hits) > 0 else 0.0
    return precision, recall


def compute_faithfulness_score(
    answer: str,
    retrieved_texts: list[str],
    expected_keywords: list[str],
    is_unanswerable: bool = False,
    was_refused: bool = False,
) -> float:
    """Compute lexical & keyword grounding faithfulness in [0.0, 1.0]."""
    if is_unanswerable:
        return 1.0 if was_refused else 0.0
    if was_refused or not retrieved_texts:
        return 0.0

    source_blob = " ".join(retrieved_texts).lower()
    source_tokens = {w for w in WORD_TOKEN_RE.findall(source_blob) if w not in STOPWORDS}
    answer_tokens = [
        w
        for w in WORD_TOKEN_RE.findall(answer.lower())
        if w not in STOPWORDS and not w.isdigit()
    ]
    if not answer_tokens:
        return 0.0

    grounded_tokens = sum(1 for w in answer_tokens if w in source_tokens)
    token_precision = grounded_tokens / len(answer_tokens)

    if expected_keywords:
        ans_lower = answer.lower()
        kw_hits = sum(1 for kw in expected_keywords if kw.lower() in ans_lower)
        kw_score = kw_hits / len(expected_keywords)
        return round(min(1.0, 0.55 * token_precision + 0.45 * kw_score), 4)

    return round(min(1.0, token_precision), 4)
