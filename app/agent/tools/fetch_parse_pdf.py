"""PDF fetcher, PyMuPDF parser (with invisible text prompt-injection filtering), S3 bundler, and FAISS indexer."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlencode

import fitz  # PyMuPDF
import httpx
from pydantic import BaseModel, Field, field_validator

from app.agent.tools.arxiv_search import ARXIV_API_URL, ArxivPaperMeta, parse_arxiv_atom_xml
from app.config import get_settings
from app.guardrails.injection import sanitize_untrusted_text
from app.observability.logger import get_logger
from app.rag.chunker import PaperChunk, chunk_paper
from app.rag.vector_store import FAISSVectorStore
from app.storage.s3 import S3Store

ARXIV_ID_VALIDATOR_RE = re.compile(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|upload-[a-zA-Z0-9_-]{3,40})$")


class FetchParsePdfInput(BaseModel):
    """Pydantic schema for fetch_and_parse_pdf tool input."""

    arxiv_id: str = Field(..., description="Valid arXiv identifier such as 2005.11401")
    raw_text_override: str | None = Field(
        default=None,
        description="Optional pre-extracted text (used for offline seed loading and testing)",
    )
    title_override: str | None = None
    authors_override: list[str] | None = None
    published_override: str | None = None

    @field_validator("arxiv_id")
    @classmethod
    def validate_arxiv_id(cls, v: str) -> str:
        clean = v.strip().replace("arXiv:", "").replace("arxiv:", "")
        if not ARXIV_ID_VALIDATOR_RE.match(clean):
            raise ValueError(f"Invalid arXiv ID format: '{v}'. Expected format like '2005.11401'.")
        return clean


def extract_safe_text_from_pdf_bytes(pdf_bytes: bytes) -> tuple[str, dict[int, str]]:
    """Extract text from PDF bytes using PyMuPDF while filtering out tiny/invisible prompt-injection spans."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_pages: list[str] = []
    page_map: dict[int, str] = {}

    for page_idx, page in enumerate(doc, start=1):
        blocks = page.get_text("dict").get("blocks", [])
        page_lines: list[str] = []

        for b in blocks:
            for line in b.get("lines", []):
                span_texts: list[str] = []
                for span in line.get("spans", []):
                    font_size = float(span.get("size", 10.0))
                    color = int(span.get("color", 0))
                    text = str(span.get("text", "")).strip()
                    # Defense against hidden white-on-white (0xFFFFFF) or microscopic (< 2pt) PDF prompt injection
                    if font_size < 2.0 or color == 0xFFFFFF:
                        continue
                    if text:
                        span_texts.append(text)
                if span_texts:
                    page_lines.append(" ".join(span_texts))

        page_text = "\n".join(page_lines).strip()
        if page_text:
            sanitized_page = sanitize_untrusted_text(page_text).cleaned_text
            full_pages.append(sanitized_page)
            page_map[page_idx] = sanitized_page

    doc.close()
    return "\n\n".join(full_pages), page_map


def _fetch_arxiv_meta_and_pdf(arxiv_id: str) -> tuple[ArxivPaperMeta, bytes]:
    params = {"id_list": arxiv_id, "max_results": 1}
    meta_url = f"{ARXIV_API_URL}?{urlencode(params)}"
    with httpx.Client(timeout=25.0, follow_redirects=True) as client:
        meta_resp = client.get(meta_url)
        meta_resp.raise_for_status()
        parsed_list = parse_arxiv_atom_xml(meta_resp.text)
        if parsed_list:
            meta = parsed_list[0]
        else:
            meta = ArxivPaperMeta(
                arxiv_id=arxiv_id,
                title=f"arXiv Paper {arxiv_id}",
                authors=["Unknown"],
                published="2024-01-01",
                summary="",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            )
        pdf_resp = client.get(meta.pdf_url)
        pdf_resp.raise_for_status()
        return meta, pdf_resp.content


def run_ingest_uploaded_pdf(
    pdf_bytes: bytes,
    filename: str,
    vector_store: FAISSVectorStore,
    title: str | None = None,
    authors: list[str] | None = None,
    s3_store: S3Store | None = None,
) -> dict[str, Any]:
    """Parse an uploaded PDF file, sanitize against prompt injection, chunk, and index in FAISS + S3."""
    import hashlib

    raw_text, page_map = extract_safe_text_from_pdf_bytes(pdf_bytes)
    if not raw_text.strip():
        raise ValueError(f"Uploaded PDF '{filename}' contains no extractable text.")

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", filename.rsplit(".", 1)[0]).strip("-").lower()[:20] or "paper"
    short_hash = hashlib.sha256(pdf_bytes).hexdigest()[:6]
    paper_id = f"upload-{slug}-{short_hash}"

    first_lines = [ln.strip() for ln in raw_text.splitlines() if len(ln.strip()) > 6]
    inferred_title = title or (first_lines[0][:120] if first_lines else filename)
    paper_authors = authors or ["Uploaded PDF Author"]
    published = "2025-01-01"

    settings = get_settings()
    s3 = s3_store or vector_store.s3_store
    chunks = chunk_paper(
        arxiv_id=paper_id,
        title=inferred_title,
        authors=paper_authors,
        published=published,
        text=raw_text,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        page_map=page_map,
    )
    bundle = {
        "arxiv_id": paper_id,
        "title": inferred_title,
        "authors": paper_authors,
        "published": published,
        "chunks": [c.to_dict() for c in chunks],
    }
    s3_key = s3.save_paper_bundle(paper_id, bundle)
    added = vector_store.add_chunks(chunks, sync_s3=settings.aws_enabled)

    get_logger().info(
        "uploaded_pdf_ingested",
        paper_id=paper_id,
        filename=filename,
        chunks=len(chunks),
    )
    return {
        "arxiv_id": paper_id,
        "title": inferred_title,
        "authors": paper_authors,
        "published": published,
        "chunks_total": len(chunks),
        "chunks_newly_indexed": added,
        "s3_key": s3_key,
        "source": "uploaded_pdf",
    }


def run_fetch_and_parse_pdf(
    raw_args: dict[str, Any],
    vector_store: FAISSVectorStore,
    s3_store: S3Store | None = None,
) -> dict[str, Any]:
    """Fetch, sanitize, chunk, store in S3 (1 compressed JSON bundle), and index in FAISS."""
    validated = FetchParsePdfInput.model_validate(raw_args)
    settings = get_settings()
    s3 = s3_store or vector_store.s3_store

    # 1. Check S3 / Local Mirror Cache first (0 arXiv bandwidth, 0 extra S3 PUTs)
    if validated.raw_text_override is None:
        cached_bundle = s3.load_paper_bundle(validated.arxiv_id)
        if cached_bundle and cached_bundle.get("chunks"):
            cached_chunks = [PaperChunk(**c) for c in cached_bundle["chunks"]]
            added = vector_store.add_chunks(cached_chunks, sync_s3=False)
            return {
                "arxiv_id": validated.arxiv_id,
                "title": cached_bundle.get("title", ""),
                "chunks_total": len(cached_chunks),
                "chunks_newly_indexed": added,
                "source": "s3_cache",
            }

    # 2. Parse from override or download live from arXiv
    page_map: dict[int, str] = {}
    if validated.raw_text_override is not None:
        raw_text = sanitize_untrusted_text(validated.raw_text_override).cleaned_text
        title = validated.title_override or f"Paper {validated.arxiv_id}"
        authors = validated.authors_override or ["Research Team"]
        published = validated.published_override or "2024-01-01"
    else:
        meta, pdf_bytes = _fetch_arxiv_meta_and_pdf(validated.arxiv_id)
        raw_text, page_map = extract_safe_text_from_pdf_bytes(pdf_bytes)
        if not raw_text.strip():
            raw_text = meta.summary
        title = meta.title
        authors = meta.authors
        published = meta.published

    chunks = chunk_paper(
        arxiv_id=validated.arxiv_id,
        title=title,
        authors=authors,
        published=published,
        text=raw_text,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        page_map=page_map,
    )

    bundle = {
        "arxiv_id": validated.arxiv_id,
        "title": title,
        "authors": authors,
        "published": published,
        "chunks": [c.to_dict() for c in chunks],
    }
    s3_key = s3.save_paper_bundle(validated.arxiv_id, bundle)
    added = vector_store.add_chunks(chunks, sync_s3=settings.aws_enabled)

    get_logger().info(
        "paper_ingested",
        arxiv_id=validated.arxiv_id,
        title=title,
        chunks=len(chunks),
        s3_key=s3_key,
    )
    return {
        "arxiv_id": validated.arxiv_id,
        "title": title,
        "authors": authors,
        "published": published,
        "chunks_total": len(chunks),
        "chunks_newly_indexed": added,
        "s3_key": s3_key,
        "source": "ingested",
    }
