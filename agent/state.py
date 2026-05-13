from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    answer_type: str
    subtasks: list[str]
    route_source: str
    classifier_confidence: float
    classifier_label: str
    plan: list[str]
    plan_index: int
    reasoning_budget: str  # "shallow" | "medium" | "deep"
    tool_calls_made: list[str]
    iteration_count: int
    max_iterations: int
    started_at: float
    max_wall_time_s: float
    confidence: str
    rewritten_query: str
    rewrite_variants: list[str]  # expanded query variants (deep budget)
    memory_context: str
    evidence_pack: dict[str, Any]  # structured evidence before generation
    critic_issues: list[str]  # content-level issues from critic pass
    retry_count: int
    needs_retry: bool
    final_answer: str
    final_answer_reviewed: bool
    validation_errors: list[str]
    answer_format_ok: bool
    metrics: dict[str, object]
