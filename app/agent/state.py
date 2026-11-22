"""LangGraph state definitions and strict Pydantic JSON schemas for tool execution and planning."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

AllowedToolName = Literal[
    "arxiv_search",
    "fetch_and_parse_pdf",
    "retrieve_chunks",
    "build_lit_matrix",
    "format_citations",
    "save_memory",
]


class PlannedToolStep(BaseModel):
    """A single planned tool invocation validated by Pydantic."""

    tool_name: AllowedToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="Required for user request")


class ExecutionPlan(BaseModel):
    """Grammar-constrained execution plan output by the Planner node."""

    intent: Literal["qa", "ingest", "lit_matrix", "citation", "memory", "search"] = "qa"
    steps: list[PlannedToolStep] = Field(..., min_length=1, max_length=4)


class LitMatrixRow(BaseModel):
    """Schema for a single row in the literature review comparison matrix."""

    arxiv_id: str
    title: str
    problem: str
    methodology: str
    datasets_or_benchmarks: str
    key_findings: str
    limitations: str
    primary_chunk_id: str


class AgentState(TypedDict, total=False):
    """LangGraph state dictionary passed across Plan-then-Execute nodes."""

    trace_id: str
    user_id: str
    query: str
    arxiv_ids: list[str]
    cached_hit: bool
    plan: dict[str, Any]
    tool_outputs: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    max_confidence: float
    low_confidence_refusal: bool
    lit_matrix_markdown: str
    lit_matrix_rows: list[dict[str, Any]]
    formatted_citations: dict[str, Any]
    draft_answer: str
    final_answer: str
    citations: list[str]
    hallucinated_citations: list[str]
    step_traces: list[dict[str, Any]]
