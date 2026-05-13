from __future__ import annotations

import os
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_bool(raw_value: str | None) -> bool | None:
    if raw_value is None:
        return None

    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def langsmith_tracing_enabled() -> bool:
    explicit = _parse_bool(os.getenv("LANGSMITH_TRACING"))
    if explicit is not None:
        return explicit
    return bool(os.getenv("LANGSMITH_API_KEY", "").strip())


def ensure_langsmith_env() -> None:
    if not langsmith_tracing_enabled():
        return

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    project = os.getenv("LANGSMITH_PROJECT", "").strip()
    if project:
        os.environ.setdefault("LANGCHAIN_PROJECT", project)


def build_trace_config(
    *,
    query: str,
    model_name: str,
    max_iterations: int,
    reasoning_budget: str | None,
    is_streaming: bool,
) -> dict[str, Any]:
    budget_label = (reasoning_budget or "auto").strip() or "auto"
    tags = [
        "langgraph-react-agent",
        f"mode:{'stream' if is_streaming else 'sync'}",
        f"budget:{budget_label}",
    ]
    metadata = {
        "query_preview": query.strip()[:120],
        "query_length": len(query),
        "model_name": model_name,
        "max_iterations": int(max_iterations),
        "reasoning_budget_requested": budget_label,
        "streaming": bool(is_streaming),
    }
    return {
        "run_name": "langgraph_agent_query",
        "tags": tags,
        "metadata": metadata,
    }
