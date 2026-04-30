from __future__ import annotations

from .intent import (
    ANSWER_COMPARISON,
    ANSWER_FACT,
    INTENT_SOTA,
)


def _normalize_entity_text(query: str) -> str:
    normalized = " ".join(query.strip().split())
    if not normalized:
        return ""

    tokens = normalized.split(" ")
    collapsed: list[str] = []
    for token in tokens:
        if collapsed and collapsed[-1].lower() == token.lower():
            continue
        collapsed.append(token)
    return " ".join(collapsed)


def rewrite_query(query: str, answer_type: str, intent: str | None = None) -> str:
    """Rewrite the query to improve tool recall without changing intent."""
    normalized = _normalize_entity_text(query)

    if answer_type == ANSWER_FACT:
        return f"{normalized} official source exact value"

    if answer_type == ANSWER_COMPARISON:
        return f"{normalized} detailed comparison differences advantages disadvantages use cases"

    if intent == INTENT_SOTA:
        return f"{normalized} latest official update"

    return normalized
