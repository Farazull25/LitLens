"""Planner node: produces a constrained JSON execution plan (1-4 steps) for the user request."""

from __future__ import annotations

from app.agent.state import AgentState, ExecutionPlan
from app.llm.base import LLMProvider
from app.llm.prompts import PLANNER_SYSTEM_PROMPT
from app.observability.logger import trace_step


def plan_execution(state: AgentState, llm: LLMProvider) -> AgentState:
    """Generate a Pydantic-validated ExecutionPlan for the current query."""
    query = state.get("query", "")
    with trace_step("planner", query_length=len(query)) as meta:
        plan_model = llm.generate_structured(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=query,
            schema_model=ExecutionPlan,
        )
        plan_dict = plan_model.model_dump()
        meta["intent"] = plan_dict["intent"]
        meta["num_steps"] = len(plan_dict["steps"])

    return {
        **state,
        "plan": plan_dict,
    }
