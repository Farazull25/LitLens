"""Map-Reduce Literature Review Comparison Matrix builder with verifiable per-row chunk citations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent.state import LitMatrixRow
from app.config import get_settings
from app.guardrails.injection import wrap_untrusted_chunk
from app.llm.base import LLMProvider
from app.llm.prompts import LIT_MATRIX_EXTRACTION_PROMPT
from app.rag.vector_store import FAISSVectorStore


class BuildLitMatrixInput(BaseModel):
    """Pydantic schema for build_lit_matrix tool input."""

    arxiv_ids: list[str] = Field(default_factory=list, description="List of arXiv IDs to compare")
    topic: str = Field(default="methodology, benchmarks, and key findings")


def _render_markdown_table(rows: list[LitMatrixRow]) -> str:
    if not rows:
        return "No indexed papers available to build a literature review matrix."

    header = (
        "| Paper (arXiv ID) | Research Problem | Methodology | Datasets / Benchmarks | Key Findings | Limitations | Verifiable Citation |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| **{r.title}** (`{r.arxiv_id}`) | {r.problem} | {r.methodology} | "
            f"{r.datasets_or_benchmarks} | {r.key_findings} | {r.limitations} | `[{r.primary_chunk_id}]` |"
        )
    return "\n".join(lines)


def run_build_lit_matrix(
    raw_args: dict[str, Any],
    vector_store: FAISSVectorStore,
    llm: LLMProvider,
) -> dict[str, Any]:
    """Run Map-Reduce extraction across selected papers to produce a grounded comparison matrix."""
    validated = BuildLitMatrixInput.model_validate(raw_args)
    settings = get_settings()

    target_ids = [aid.strip() for aid in validated.arxiv_ids if aid.strip()]
    if not target_ids:
        # Default to up to max_papers_in_matrix currently indexed papers
        indexed_papers = vector_store.list_papers()
        target_ids = [p["arxiv_id"] for p in indexed_papers[: settings.max_papers_in_matrix]]
    else:
        target_ids = target_ids[: settings.max_papers_in_matrix]

    rows: list[LitMatrixRow] = []
    for aid in target_ids:
        paper_hits = vector_store.search(
            query=f"{validated.topic} abstract methodology experiments results limitations",
            top_k=3,
            arxiv_ids=[aid],
        )
        if not paper_hits:
            continue

        valid_chunk_ids = {h.chunk.chunk_id for h in paper_hits}
        default_cid = paper_hits[0].chunk.chunk_id
        title = paper_hits[0].chunk.title

        xml_context = "\n\n".join(
            wrap_untrusted_chunk(
                chunk_id=h.chunk.chunk_id,
                title=h.chunk.title,
                section=h.chunk.section,
                text=h.chunk.text,
            )
            for h in paper_hits
        )
        row = llm.generate_structured(
            system_prompt=LIT_MATRIX_EXTRACTION_PROMPT.format(arxiv_id=aid),
            user_prompt=xml_context,
            schema_model=LitMatrixRow,
        )
        # Enforce ground-truth arxiv_id, title, and verified chunk_id
        verified_cid = row.primary_chunk_id if row.primary_chunk_id in valid_chunk_ids else default_cid
        rows.append(
            LitMatrixRow(
                arxiv_id=aid,
                title=title,
                problem=row.problem,
                methodology=row.methodology,
                datasets_or_benchmarks=row.datasets_or_benchmarks,
                key_findings=row.key_findings,
                limitations=row.limitations,
                primary_chunk_id=verified_cid,
            )
        )

    markdown = _render_markdown_table(rows)
    return {
        "paper_count": len(rows),
        "rows": [r.model_dump() for r in rows],
        "markdown": markdown,
        "citations": [r.primary_chunk_id for r in rows],
    }
