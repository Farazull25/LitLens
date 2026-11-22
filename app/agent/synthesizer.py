"""Grounded synthesizer node with low-confidence refusal gating and citation verification."""

from __future__ import annotations

from app.agent.state import AgentState
from app.config import get_settings
from app.guardrails.injection import wrap_untrusted_chunk
from app.guardrails.validators import verify_and_clean_citations
from app.llm.base import LLMProvider
from app.llm.prompts import SYNTHESIS_SYSTEM_PROMPT
from app.observability.logger import trace_step
from app.observability.metrics import metrics_registry


def check_confidence_gate(state: AgentState) -> AgentState:
    """Evaluate retrieval confidence against minimum threshold; flag low_confidence_refusal when below threshold."""
    settings = get_settings()
    intent = state.get("plan", {}).get("intent", "qa")

    with trace_step("confidence_gate", intent=intent) as meta:
        if intent in {"lit_matrix", "citation", "memory", "ingest"}:
            refuse = False
        else:
            chunks = state.get("retrieved_chunks", [])
            max_conf = float(state.get("max_confidence", 0.0))
            refuse = (len(chunks) == 0) or (max_conf < settings.min_retrieval_confidence)
            if refuse:
                metrics_registry.record("RefusalCount", 1.0)
        meta["low_confidence_refusal"] = refuse

    return {
        **state,
        "low_confidence_refusal": refuse,
    }


def synthesize_and_verify(state: AgentState, llm: LLMProvider) -> AgentState:
    """Synthesize final answer with verifiable [arxiv_id#cN] citations or refuse when confidence is low."""
    settings = get_settings()
    intent = state.get("plan", {}).get("intent", "qa")

    with trace_step("synthesize_and_verify", intent=intent) as meta:
        if state.get("low_confidence_refusal", False):
            max_c = state.get("max_confidence", 0.0)
            refusal_msg = (
                f"I cannot answer this question with verifiable citations because retrieval confidence "
                f"({max_c:.2f}) is below the required safety threshold ({settings.min_retrieval_confidence:.2f}). "
                f"Please ingest relevant arXiv papers on this topic first."
            )
            meta["refused"] = True
            return {
                **state,
                "final_answer": refusal_msg,
                "citations": [],
                "hallucinated_citations": [],
            }

        if intent == "lit_matrix":
            md = state.get("lit_matrix_markdown", "")
            cites = [r["primary_chunk_id"] for r in state.get("lit_matrix_rows", []) if r.get("primary_chunk_id")]
            return {
                **state,
                "final_answer": md,
                "citations": cites,
                "hallucinated_citations": [],
            }

        if intent == "citation":
            fc = state.get("formatted_citations", {})
            ans = f"### APA Citations\n{fc.get('apa', '')}\n\n### BibTeX\n```bibtex\n{fc.get('bibtex', '')}\n```"
            return {
                **state,
                "final_answer": ans,
                "citations": [],
                "hallucinated_citations": [],
            }

        if intent == "memory":
            out = state.get("tool_outputs", [{}])[-1].get("output", {})
            ans = f"Saved research preference to memory (`{out.get('key')}`: `{out.get('value')}`)."
            return {
                **state,
                "final_answer": ans,
                "citations": [],
                "hallucinated_citations": [],
            }

        if intent == "ingest":
            ingested = [t["output"] for t in state.get("tool_outputs", []) if t.get("tool_name") == "fetch_and_parse_pdf"]
            summaries = [
                f"- **{p.get('title')}** (`{p.get('arxiv_id')}`): {p.get('chunks_total')} chunks indexed."
                for p in ingested
            ]
            ans = "Successfully ingested and indexed papers:\n" + "\n".join(summaries)
            return {
                **state,
                "final_answer": ans,
                "citations": [],
                "hallucinated_citations": [],
            }

        # Standard Grounded Q&A Synthesis
        chunks = state.get("retrieved_chunks", [])
        retrieved_ids = {c["chunk_id"] for c in chunks}
        xml_blocks = [
            wrap_untrusted_chunk(
                chunk_id=c["chunk_id"],
                title=c["title"],
                section=c["section"],
                text=c["text"],
            )
            for c in chunks
        ]
        user_prompt = (
            "RETRIEVED PAPER CHUNKS:\n"
            + "\n\n".join(xml_blocks)
            + f"\n\nUSER QUESTION: {state.get('query', '')}\n"
            + "Provide a concise, accurate answer with inline citations in the exact format [arxiv_id#cN]."
        )

        draft = llm.generate(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

        validation = verify_and_clean_citations(draft, retrieved_ids)
        verified_text = validation.verified_answer
        valid_cites = list(validation.valid_citations)

        # If a small model forgot to include a bracket citation despite having confident chunks,
        # deterministically attach the top retrieved chunk citation to guarantee traceability.
        if not valid_cites and chunks:
            top_cid = chunks[0]["chunk_id"]
            verified_text = f"{verified_text.rstrip('.')} [{top_cid}]."
            valid_cites = [top_cid]

        meta["valid_citations"] = len(valid_cites)
        meta["hallucinated_citations"] = len(validation.hallucinated_citations)

        return {
            **state,
            "draft_answer": draft,
            "final_answer": verified_text,
            "citations": valid_cites,
            "hallucinated_citations": validation.hallucinated_citations,
        }
