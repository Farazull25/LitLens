"""End-to-end evaluation harness scoring the 30-question Golden Dataset."""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from app.agent.graph import LitLensCopilotAgent
from app.llm.hf_fallback_provider import get_llm_provider
from app.rag.chunker import chunk_paper
from app.rag.embeddings import get_embedder
from app.rag.vector_store import FAISSVectorStore
from app.storage.dynamodb import DynamoDBStore
from app.storage.s3 import S3Store

from eval.metrics import (
    SampleEvalScore,
    compute_citation_accuracy,
    compute_faithfulness_score,
    compute_retrieval_hit_and_mrr,
)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_eval_vector_store(
    dataset: dict[str, Any],
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    force_hash_embeddings: bool = True,
) -> FAISSVectorStore:
    """Seed an isolated FAISS vector store with the golden dataset papers at the requested chunk size."""
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"litlens_eval_c{chunk_size_tokens}_"))
    embedder = get_embedder(force_hash=force_hash_embeddings)
    store = FAISSVectorStore(embedder=embedder, index_dir=tmp_dir, s3_store=S3Store())

    all_chunks = []
    for paper in dataset.get("seed_papers", []):
        chunks = chunk_paper(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            authors=paper["authors"],
            published=paper["published"],
            text=paper["text"],
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=min(chunk_overlap_tokens, chunk_size_tokens // 4),
        )
        all_chunks.extend(chunks)

    store.add_chunks(all_chunks, sync_s3=False)
    return store


def run_evaluation(
    chunk_size_tokens: int = 512,
    model_size: str = "1.5B",
    force_hash_embeddings: bool = True,
    persist_to_dynamodb: bool = True,
) -> dict[str, Any]:
    """Run all 30 golden questions and compute aggregate retrieval, citation, and faithfulness metrics."""
    dataset = load_golden_dataset()
    store = build_eval_vector_store(
        dataset=dataset,
        chunk_size_tokens=chunk_size_tokens,
        force_hash_embeddings=force_hash_embeddings,
    )
    ddb = DynamoDBStore()
    llm = get_llm_provider(model_size=model_size)
    agent = LitLensCopilotAgent(vector_store=store, ddb_store=ddb, llm=llm)

    sample_scores: list[SampleEvalScore] = []
    questions = dataset.get("questions", [])

    for item in questions:
        qid = item["id"]
        q_text = item["question"]
        expected_aid = item.get("expected_arxiv_id")
        expected_kws = item.get("expected_keywords", [])
        is_unanswerable = bool(item.get("is_unanswerable", False))

        t0 = time.perf_counter()
        state = agent.run(query=q_text, use_cache=False)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        retrieved_chunks = state.get("retrieved_chunks", [])
        was_refused = bool(state.get("low_confidence_refusal", False))
        retrieved_aids = [c["arxiv_id"] for c in retrieved_chunks]
        retrieved_cids = {c["chunk_id"] for c in retrieved_chunks}
        retrieved_texts = [c["text"] for c in retrieved_chunks]
        cited_cids = state.get("citations", [])
        answer = state.get("final_answer", "")

        if is_unanswerable:
            hit = 1.0 if was_refused else 0.0
            mrr = 1.0 if was_refused else 0.0
        else:
            hit, mrr = compute_retrieval_hit_and_mrr(retrieved_aids, expected_aid)

        c_prec, c_rec = compute_citation_accuracy(cited_cids, retrieved_cids, expected_aid)
        faith = compute_faithfulness_score(
            answer=answer,
            retrieved_texts=retrieved_texts,
            expected_keywords=expected_kws,
            is_unanswerable=is_unanswerable,
            was_refused=was_refused,
        )

        sample_scores.append(
            SampleEvalScore(
                question_id=qid,
                retrieval_hit=hit,
                reciprocal_rank=mrr,
                citation_precision=c_prec,
                citation_recall=c_rec,
                faithfulness=faith,
                correct_refusal=(was_refused == is_unanswerable),
                latency_ms=latency_ms,
            )
        )

    n = max(len(sample_scores), 1)
    summary = {
        "run_id": uuid.uuid4().hex[:10],
        "model_size": model_size,
        "chunk_size_tokens": chunk_size_tokens,
        "num_questions": len(sample_scores),
        "retrieval_hit_rate": round(sum(s.retrieval_hit for s in sample_scores) / n, 4),
        "mrr": round(sum(s.reciprocal_rank for s in sample_scores) / n, 4),
        "citation_precision": round(sum(s.citation_precision for s in sample_scores) / n, 4),
        "citation_recall": round(sum(s.citation_recall for s in sample_scores) / n, 4),
        "faithfulness": round(sum(s.faithfulness for s in sample_scores) / n, 4),
        "abstention_accuracy": round(sum(1.0 for s in sample_scores if s.correct_refusal) / n, 4),
        "avg_latency_ms": round(sum(s.latency_ms for s in sample_scores) / n, 2),
        "samples": [s.to_dict() for s in sample_scores],
    }

    if persist_to_dynamodb:
        ddb.log_eval_run(
            run_id=summary["run_id"],
            metrics={k: v for k, v in summary.items() if k != "samples"},
        )

    return summary


if __name__ == "__main__":
    res = run_evaluation()
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=2))
