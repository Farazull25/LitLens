"""Citation formatting tool generating BibTeX and APA citations from indexed arXiv metadata."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.vector_store import FAISSVectorStore


class FormatCitationsInput(BaseModel):
    """Pydantic schema for format_citations tool input."""

    arxiv_ids: list[str] = Field(default_factory=list)
    style: Literal["bibtex", "apa", "both"] = "both"


def _to_bibtex(arxiv_id: str, title: str, authors: str, year: str) -> str:
    first_author_last = re.sub(r"[^a-zA-Z]", "", authors.split(",")[0].split()[-1]).lower() or "author"
    cite_key = f"{first_author_last}{year}_{arxiv_id.replace('.', '_')}"
    bibtex_authors = " and ".join(a.strip() for a in authors.split(",") if a.strip())
    return (
        f"@article{{{cite_key},\n"
        f"  title={{{title}}},\n"
        f"  author={{{bibtex_authors}}},\n"
        f"  journal={{arXiv preprint arXiv:{arxiv_id}}},\n"
        f"  year={{{year}}},\n"
        f"  url={{https://arxiv.org/abs/{arxiv_id}}}\n"
        f"}}"
    )


def _to_apa(arxiv_id: str, title: str, authors: str, year: str) -> str:
    return f"{authors} ({year}). {title}. arXiv preprint arXiv:{arxiv_id}. https://arxiv.org/abs/{arxiv_id}"


def run_format_citations(
    raw_args: dict[str, Any],
    vector_store: FAISSVectorStore,
) -> dict[str, Any]:
    """Generate BibTeX and APA references for specified or indexed arXiv papers."""
    validated = FormatCitationsInput.model_validate(raw_args)
    indexed = {p["arxiv_id"]: p for p in vector_store.list_papers()}
    target_ids = validated.arxiv_ids or list(indexed.keys())

    bibtex_entries: list[str] = []
    apa_entries: list[str] = []

    for aid in target_ids:
        meta = indexed.get(
            aid,
            {
                "arxiv_id": aid,
                "title": f"arXiv Paper {aid}",
                "authors": "arXiv Author",
                "published": "2024-01-01",
            },
        )
        year = meta["published"][:4] if len(meta.get("published", "")) >= 4 else "2024"
        if validated.style in ("bibtex", "both"):
            bibtex_entries.append(_to_bibtex(aid, meta["title"], meta["authors"], year))
        if validated.style in ("apa", "both"):
            apa_entries.append(_to_apa(aid, meta["title"], meta["authors"], year))

    return {
        "count": len(target_ids),
        "bibtex": "\n\n".join(bibtex_entries),
        "apa": "\n".join(apa_entries),
    }
