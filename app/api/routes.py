"""FastAPI API routes for chat, paper ingestion, literature matrix generation, digest subscription, and evaluation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from eval.benchmark_grid import run_benchmark_grid
from eval.runner import load_golden_dataset
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import LitLensCopilotAgent
from app.agent.tools.build_lit_matrix import run_build_lit_matrix
from app.agent.tools.fetch_parse_pdf import run_fetch_and_parse_pdf
from app.observability.metrics import metrics_registry

router = APIRouter(prefix="/api", tags=["LitLens Copilot API"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    user_id: str = Field(default="default_researcher")
    arxiv_ids: list[str] = Field(default_factory=list)
    use_cache: bool = True


class IngestRequest(BaseModel):
    arxiv_id: str = Field(..., description="arXiv ID e.g. 2401.10001")
    raw_text_override: str | None = None
    title_override: str | None = None


class MatrixRequest(BaseModel):
    arxiv_ids: list[str] = Field(default_factory=list)
    topic: str = Field(default="methodology, benchmarks, and key findings")


class SubscribeRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=120)
    topic: str = Field(..., min_length=2, max_length=100)


@lru_cache(maxsize=1)
def get_copilot_agent() -> LitLensCopilotAgent:
    """Initialize singleton Copilot Agent and pre-seed the golden corpus if the index is empty."""
    agent = LitLensCopilotAgent()
    if not agent.vector_store.load_local_or_s3():
        dataset = load_golden_dataset()
        for paper in dataset.get("seed_papers", []):
            run_fetch_and_parse_pdf(
                {
                    "arxiv_id": paper["arxiv_id"],
                    "raw_text_override": paper["text"],
                    "title_override": paper["title"],
                    "authors_override": paper["authors"],
                    "published_override": paper["published"],
                },
                vector_store=agent.vector_store,
            )
    return agent


@router.get("/health")
def health_check() -> dict[str, Any]:
    agent = get_copilot_agent()
    return {
        "status": "healthy",
        "indexed_papers": len(agent.vector_store.list_papers()),
        "indexed_chunks": len(agent.vector_store.chunks),
        "metrics": metrics_registry.snapshot(),
    }


@router.get("/papers")
def list_indexed_papers() -> dict[str, Any]:
    agent = get_copilot_agent()
    papers = agent.vector_store.list_papers()
    return {"count": len(papers), "papers": papers}


@router.post("/chat")
def chat_endpoint(req: ChatRequest) -> dict[str, Any]:
    agent = get_copilot_agent()
    try:
        state = agent.run(
            query=req.query,
            user_id=req.user_id,
            arxiv_ids=req.arxiv_ids or None,
            use_cache=req.use_cache,
        )
        return {
            "trace_id": state.get("trace_id"),
            "answer": state.get("final_answer"),
            "citations": state.get("citations", []),
            "max_confidence": state.get("max_confidence", 0.0),
            "low_confidence_refusal": state.get("low_confidence_refusal", False),
            "cached_hit": state.get("cached_hit", False),
            "plan": state.get("plan"),
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "step_traces": state.get("step_traces", []),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ingest")
def ingest_endpoint(req: IngestRequest) -> dict[str, Any]:
    agent = get_copilot_agent()
    try:
        result = run_fetch_and_parse_pdf(
            {
                "arxiv_id": req.arxiv_id,
                "raw_text_override": req.raw_text_override,
                "title_override": req.title_override,
            },
            vector_store=agent.vector_store,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to ingest arXiv paper: {exc}") from exc


@router.post("/matrix")
def matrix_endpoint(req: MatrixRequest) -> dict[str, Any]:
    agent = get_copilot_agent()
    result = run_build_lit_matrix(
        {"arxiv_ids": req.arxiv_ids, "topic": req.topic},
        vector_store=agent.vector_store,
        llm=agent.llm,
    )
    return result


@router.post("/subscribe")
def subscribe_endpoint(req: SubscribeRequest) -> dict[str, Any]:
    if "@" not in req.email or "." not in req.email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid email address format.")
    agent = get_copilot_agent()
    ok = agent.ddb_store.subscribe_topic(email=req.email, topic=req.topic)
    return {
        "subscribed": ok,
        "email": req.email,
        "topic": req.topic,
        "subscriptions": agent.ddb_store.list_subscriptions(),
    }


@router.get("/eval/grid")
def eval_grid_endpoint() -> dict[str, Any]:
    return run_benchmark_grid()
