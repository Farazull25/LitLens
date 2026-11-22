"""UI formatting helpers for citation badges, retrieved chunk cards, and telemetry summaries."""

from __future__ import annotations

from typing import Any


def format_chunks_inspector_markdown(
    chunks: list[dict[str, Any]],
    citations: list[str],
    max_confidence: float,
    cached_hit: bool,
) -> str:
    """Format retrieved chunks and verification status as rich Markdown for the Gradio side panel."""
    badge = "🟢 **VERIFIED GROUNDED**" if citations else "🟡 **ABSTAINED / NO CITATIONS**"
    cache_tag = "⚡ *DynamoDB Cache Hit*" if cached_hit else "🧠 *Live LangGraph Execution*"
    lines = [
        f"### Grounding Status: {badge}",
        f"- **Max Hybrid Confidence**: `{max_confidence:.3f}` | **Execution**: {cache_tag}",
        f"- **Verified Inline Citations**: `{', '.join(citations) if citations else 'None'}`",
        "---",
        "#### Retrieved FAISS + BM25 Evidence Chunks",
    ]
    if not chunks:
        lines.append("*No chunks retrieved.*")
        return "\n".join(lines)

    for c in chunks:
        cid = c.get("chunk_id", "")
        is_cited = " **(CITED)**" if cid in citations else ""
        lines.append(
            f"**`[{cid}]`{is_cited}** — *{c.get('title', '')}* (Section: `{c.get('section', '')}`, "
            f"p.{c.get('page_number', 1)}, Score: `{c.get('hybrid_score', 0.0):.3f}`)\n"
            f"> {c.get('text', '')[:380]}..."
        )
    return "\n\n".join(lines)
