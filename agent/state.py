from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    answer_type: str
    route_source: str
    classifier_confidence: float
    classifier_label: str
    plan: list[str]
    plan_index: int
    depth_mode: str
    tool_calls_made: list[str]
    iteration_count: int
    max_iterations: int
    confidence: str
    rewritten_query: str
    memory_context: str
    retry_count: int
    needs_retry: bool
    final_answer: str
    final_answer_reviewed: bool
    validation_errors: list[str]
    answer_format_ok: bool
    metrics: dict[str, object]
