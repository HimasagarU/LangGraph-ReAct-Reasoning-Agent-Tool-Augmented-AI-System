from __future__ import annotations

import re
from typing import Final

from .intent import (
    ANSWER_CALCULATION,
    ANSWER_AMBIGUOUS,
    ANSWER_COMPARISON,
    ANSWER_EXPLANATION,
    ANSWER_FACT,
    INTENT_COMPARATIVE,
    INTENT_EXPLANATORY,
    INTENT_SOTA,
    INTENT_TECHNICAL,
)


# ── synonym / perspective tables for heuristic rewrite ────────────────────────

_SYNONYM_MAP: Final[dict[str, list[str]]] = {
    "difference": ["differences", "distinction", "contrast"],
    "compare": ["comparison", "versus", "tradeoffs"],
    "explain": ["how does", "what is the mechanism behind", "intuition for"],
    "best": ["recommended", "preferred", "state-of-the-art"],
    "latest": ["most recent", "current", "2025-2026"],
    "advantage": ["benefit", "strength", "pro"],
    "disadvantage": ["drawback", "weakness", "con"],
    "use case": ["application", "scenario", "when to use"],
}

_PERSPECTIVE_SUFFIXES: Final[list[str]] = [
    "with practical examples",
    "key tradeoffs and limitations",
    "compared to alternatives",
    "step by step explanation",
]


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


def rewrite_query(
    query: str,
    answer_type: str,
    intent: str | None = None,
    budget: str = "shallow",
) -> str:
    """Rewrite the query to improve tool recall without changing intent.

    For shallow budget, only normalizes. For medium/deep, appends
    answer-type-specific expansion terms.
    """
    normalized = _normalize_entity_text(query)

    if budget == "shallow":
        return normalized

    # Skip expansion for types that need precision, not recall
    if answer_type in {ANSWER_CALCULATION, ANSWER_AMBIGUOUS}:
        return normalized

    if answer_type == ANSWER_FACT:
        return f"{normalized} official source exact value"

    if answer_type == ANSWER_COMPARISON or intent == INTENT_COMPARATIVE:
        return f"{normalized} differences advantages disadvantages use cases"

    if intent == INTENT_SOTA:
        return f"{normalized} latest official update"

    if answer_type == ANSWER_EXPLANATION or intent in {INTENT_EXPLANATORY, INTENT_TECHNICAL}:
        return f"{normalized} explanation mechanism intuition"

    return normalized


def generate_rewrite_variants(
    query: str,
    answer_type: str,
    intent: str | None = None,
) -> list[str]:
    """Generate 2-4 heuristic query variants for deep budget evidence gathering.

    Uses synonym swaps, specificity boosts, and perspective shifts.
    Does NOT call an LLM — purely deterministic.
    """
    normalized = _normalize_entity_text(query)
    if not normalized:
        return []

    # Skip variant generation for types that don't benefit
    if answer_type in {ANSWER_FACT, ANSWER_CALCULATION, ANSWER_AMBIGUOUS}:
        return []

    lowered = normalized.lower()
    variants: list[str] = []

    # 1) Synonym swap: replace the first matching term
    for term, synonyms in _SYNONYM_MAP.items():
        if term in lowered:
            replacement = synonyms[0]
            swapped = re.sub(re.escape(term), replacement, lowered, count=1)
            variant = " ".join(swapped.split()).strip()
            if variant and variant != lowered:
                variants.append(variant)
            break  # one swap per variant set

    # 2) Perspective shift: append a context-appropriate suffix
    if answer_type == ANSWER_COMPARISON or intent == INTENT_COMPARATIVE:
        variants.append(f"{normalized} key tradeoffs and limitations")
    elif intent == INTENT_SOTA:
        variants.append(f"{normalized} recent breakthroughs and benchmarks")
    elif answer_type == ANSWER_EXPLANATION or intent in {INTENT_EXPLANATORY, INTENT_TECHNICAL}:
        variants.append(f"{normalized} with practical examples")
    else:
        variants.append(f"{normalized} {_PERSPECTIVE_SUFFIXES[0]}")

    # 3) Specificity boost: make the query more targeted
    if intent == INTENT_TECHNICAL:
        variants.append(f"{normalized} algorithm implementation details")
    elif intent == INTENT_COMPARATIVE:
        variants.append(f"{normalized} performance benchmarks comparison table")
    elif answer_type == ANSWER_EXPLANATION:
        variants.append(f"{normalized} step by step breakdown with example")

    # Deduplicate and cap at 4
    seen: set[str] = {lowered}
    unique_variants: list[str] = []
    for v in variants:
        v_lower = v.lower().strip()
        if v_lower not in seen and v_lower:
            seen.add(v_lower)
            unique_variants.append(v.strip())
    return unique_variants[:4]
