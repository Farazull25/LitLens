"""LangGraph Plan-then-Execute state machine orchestrating LitLens."""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph

from app.agent.executor import execute_plan_steps
from app.agent.planner import plan_execution
from app.agent.state import AgentState
from app.agent.synthesizer import check_confidence_gate, synthesize_and_verify
from app.guardrails.validators import validate_user_query
from app.llm.base import LLMProvider
from app.llm.hf_fallback_provider import get_llm_provider
from app.observability.logger import get_step_traces, start_trace
from app.observability.metrics import metrics_registry
from app.rag.vector_store import FAISSVectorStore
from app.storage.dynamodb import DynamoDBStore


class LitLensCopilotAgent:
    """Production LangGraph Plan-then-Execute Copilot agent with caching, guardrails, and citation verification."""

    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        ddb_store: DynamoDBStore | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.vector_store = vector_store or FAISSVectorStore()
        self.ddb_store = ddb_store or DynamoDBStore()
        self.llm = llm or get_llm_provider()
        self._compiled_graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", lambda s: plan_execution(s, self.llm))
        workflow.add_node(
            "executor",
            lambda s: execute_plan_steps(s, self.vector_store, self.ddb_store, self.llm),
        )
        workflow.add_node("confidence_gate", check_confidence_gate)
        workflow.add_node("synthesizer", lambda s: synthesize_and_verify(s, self.llm))

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "confidence_gate")
        workflow.add_edge("confidence_gate", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def run(
        self,
        query: str,
        user_id: str = "default_researcher",
        arxiv_ids: list[str] | None = None,
        use_cache: bool = True,
    ) -> AgentState:
        """Execute the Plan-then-Execute workflow with input validation, caching, and latency metrics."""
        t0 = time.perf_counter()
        trace_id = start_trace()
        clean_query = validate_user_query(query)

        cache_key = f"{clean_query}::{','.join(sorted(arxiv_ids or []))}"
        if use_cache:
            cached = self.ddb_store.get_cached_response(cache_key)
            if cached:
                metrics_registry.record("CacheHitCount", 1.0)
                latency = round((time.perf_counter() - t0) * 1000.0, 2)
                metrics_registry.record("QueryLatencyMs", latency, unit="Milliseconds")
                return {
                    **cached,
                    "trace_id": trace_id,
                    "cached_hit": True,
                }

        initial_state: AgentState = {
            "trace_id": trace_id,
            "user_id": user_id,
            "query": clean_query,
            "arxiv_ids": arxiv_ids or [],
            "cached_hit": False,
        }

        final_state: AgentState = self._compiled_graph.invoke(initial_state)
        final_state["step_traces"] = get_step_traces()

        latency = round((time.perf_counter() - t0) * 1000.0, 2)
        metrics_registry.record("QueryLatencyMs", latency, unit="Milliseconds")

        if use_cache and not final_state.get("low_confidence_refusal", False):
            cacheable = {
                "query": clean_query,
                "plan": final_state.get("plan"),
                "retrieved_chunks": final_state.get("retrieved_chunks", []),
                "max_confidence": final_state.get("max_confidence", 0.0),
                "low_confidence_refusal": False,
                "final_answer": final_state.get("final_answer", ""),
                "citations": final_state.get("citations", []),
                "hallucinated_citations": final_state.get("hallucinated_citations", []),
                "lit_matrix_markdown": final_state.get("lit_matrix_markdown", ""),
                "lit_matrix_rows": final_state.get("lit_matrix_rows", []),
            }
            self.ddb_store.set_cached_response(cache_key, cacheable)

        return final_state
