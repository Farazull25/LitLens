"""Unit tests for section-aware chunking, bibliography stripping, FAISS vector store, and RRF hybrid search."""

from __future__ import annotations

from app.rag.chunker import chunk_paper, split_into_sections, strip_bibliography
from app.rag.vector_store import FAISSVectorStore


def test_strip_bibliography_removes_trailing_references() -> None:
    sample = (
        "Abstract\nThis paper introduces a fast retrieval method for scientific corpora.\n\n"
        "Methodology\nWe combine dense vectors with lexical BM25 scoring and RRF rank fusion.\n\n"
        "Results\nOur experiments demonstrate a 14.2% recall improvement across benchmarks.\n\n"
        "References\n[1] Doe et al., 2020. Unrelated paper title that should not be indexed."
    )
    stripped = strip_bibliography(sample)
    assert "14.2% recall improvement" in stripped
    assert "Unrelated paper title" not in stripped


def test_section_aware_chunking_assigns_section_and_chunk_ids() -> None:
    sample = (
        "Abstract\nShort summary of the architecture.\n\n"
        "Methodology\nDetailed description of the 384-dimensional local encoder and FAISS index.\n\n"
        "Results\nQuantitative evaluation on SciFact and HotpotQA."
    )
    sections = split_into_sections(sample)
    section_names = [s[0] for s in sections]
    assert "Abstract" in section_names
    assert "Methodology" in section_names
    assert "Results" in section_names

    chunks = chunk_paper(
        arxiv_id="2401.10001",
        title="Hybrid RRF Architecture",
        authors=["A. Chen"],
        published="2024-01-15",
        text=sample,
        chunk_size_tokens=128,
        chunk_overlap_tokens=16,
    )
    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "2401.10001#c0"
    assert chunks[0].section == "Abstract"


def test_faiss_vector_store_hybrid_search_and_persistence(temp_vector_store: FAISSVectorStore) -> None:
    c1 = chunk_paper(
        arxiv_id="2401.10001",
        title="Hybrid RRF Retrieval",
        authors=["A. Chen"],
        published="2024-01-15",
        text="Abstract\nReciprocal Rank Fusion with constant k=60 merges dense and BM25 scores.",
    )
    c2 = chunk_paper(
        arxiv_id="2401.10002",
        title="Grammar Constrained Planning",
        authors=["S. Rossi"],
        published="2024-02-10",
        text="Abstract\nGBNF context-free grammars enforce 100% valid JSON plans on Q4_K_M models.",
    )
    added = temp_vector_store.add_chunks(c1 + c2)
    assert added == len(c1) + len(c2)
    # Duplicate addition should be ignored
    assert temp_vector_store.add_chunks(c1) == 0

    hits = temp_vector_store.search("What constant k is used in Reciprocal Rank Fusion?", top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk.arxiv_id == "2401.10001"
    assert hits[0].hybrid_score > hits[1].hybrid_score

    # Verify save and reload from local disk
    reloaded = FAISSVectorStore(
        embedder=temp_vector_store.embedder,
        index_dir=temp_vector_store.index_dir,
        s3_store=temp_vector_store.s3_store,
    )
    assert reloaded.load_local_or_s3() is True
    assert len(reloaded.chunks) == len(c1) + len(c2)
