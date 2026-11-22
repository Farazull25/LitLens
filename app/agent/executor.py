"""Deterministic Tool Executor node: executes planned tool calls with Pydantic validation."""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState, ExecutionPlan
from app.agent.tools.arxiv_search import run_arxiv_search
from app.agent.tools.build_lit_matrix import run_build_lit_matrix
from app.agent.tools.fetch_parse_pdf import run_fetch_and_parse_pdf
from app.agent.tools.format_citations import run_format_citations
from app.agent.tools.retrieve_chunks import run_retrieve_chunks
from app.agent.tools.save_memory import run_save_memory
from app.llm.base import LLMProvider
from app.observability.logger import trace_step
from app.rag.vector_store import FAISSVectorStore
from app.storage.dynamodb import DynamoDBStore


def execute_plan_steps(
    state: AgentState,
    vector_store: FAISSVectorStore,
    ddb_store: DynamoDBStore,
    llm: LLMProvider,
) -> AgentState:
    """Execute each planned step sequentially, collecting tool outputs and retrieved chunks."""
    plan = ExecutionPlan.model_validate(state["plan"])
    tool_outputs: list[dict[str, Any]] = []
    retrieved_chunks: list[dict[str, Any]] = list(state.get("retrieved_chunks", []))
    max_conf: float = float(state.get("max_confidence", 0.0))
    lit_matrix_md: str = state.get("lit_matrix_markdown", "")
    lit_matrix_rows: list[dict[str, Any]] = list(state.get("lit_matrix_rows", []))
    formatted_citations: dict[str, Any] = dict(state.get("formatted_citations", {}))
    filter_ids = state.get("arxiv_ids") or None

    for idx, step in enumerate(plan.steps):
        with trace_step(f"tool:{step.tool_name}", step_index=idx) as meta:
            args = dict(step.arguments)
            if step.tool_name == "arxiv_search":
                args.setdefault("query", state["query"])
                out = run_arxiv_search(args)
            elif step.tool_name == "fetch_and_parse_pdf":
                out = run_fetch_and_parse_pdf(args, vector_store=vector_store)
            elif step.tool_name == "retrieve_chunks":
                args.setdefault("query", state["query"])
                if filter_ids and not args.get("arxiv_ids"):
                    args["arxiv_ids"] = filter_ids
                out = run_retrieve_chunks(args, vector_store=vector_store)
                retrieved_chunks = out["chunks"]
                max_conf = float(out["max_confidence"])
                meta["max_confidence"] = max_conf
                meta["chunk_count"] = len(retrieved_chunks)
            elif step.tool_name == "build_lit_matrix":
                if filter_ids and not args.get("arxiv_ids"):
                    args["arxiv_ids"] = filter_ids
                out = run_build_lit_matrix(args, vector_store=vector_store, llm=llm)
                lit_matrix_md = out["markdown"]
                lit_matrix_rows = out["rows"]
                max_conf = 1.0 if lit_matrix_rows else 0.0
            elif step.tool_name == "format_citations":
                if filter_ids and not args.get("arxiv_ids"):
                    args["arxiv_ids"] = filter_ids
                out = run_format_citations(args, vector_store=vector_store)
                formatted_citations = out
                max_conf = 1.0
            elif step.tool_name == "save_memory":
                args.setdefault("user_id", state.get("user_id", "default_researcher"))
                out = run_save_memory(args, ddb_store=ddb_store)
                max_conf = 1.0
            else:
                raise ValueError(f"Unsupported tool: {step.tool_name}")

            tool_outputs.append({"tool_name": step.tool_name, "output": out})

    return {
        **state,
        "tool_outputs": tool_outputs,
        "retrieved_chunks": retrieved_chunks,
        "max_confidence": max_conf,
        "lit_matrix_markdown": lit_matrix_md,
        "lit_matrix_rows": lit_matrix_rows,
        "formatted_citations": formatted_citations,
    }
