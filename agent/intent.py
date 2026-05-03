from __future__ import annotations

import re
import os
import json
import pathlib
from typing import Final, Any

try:
    from .classifier import IntentClassifier
except Exception:
    IntentClassifier = None  # optional dependency at runtime

# instantiate classifier once (if model file exists)
_CLASSIFIER: IntentClassifier | None = None
if IntentClassifier is not None:
    try:
        _CLASSIFIER = IntentClassifier()
    except Exception:
        _CLASSIFIER = None

# simple JSONL logger for classifier decisions
_LOG_PATH = pathlib.Path(__file__).parent / "intent_logs.jsonl"

def _log_classification(query: str, predicted: str | None, confidence: float | None, fallback: str | None = None) -> None:
    entry = {
        "query": query,
        "predicted": predicted,
        "confidence": confidence,
        "fallback": fallback,
    }
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

INTENT_EXPLANATORY = "explanatory"
INTENT_SOTA = "sota"
INTENT_COMPARATIVE = "comparative"
INTENT_TECHNICAL = "technical"
INTENT_DISCOVERY = "discovery"

ANSWER_FACT = "fact"
ANSWER_LIST = "list"
ANSWER_EXPLANATION = "explanation"
ANSWER_COMPARISON = "comparison"
ANSWER_CALCULATION = "calculation"
ANSWER_AMBIGUOUS = "ambiguous"

STOPWORDS: Final[set[str]] = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "show",
    "tell",
    "that",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
    "would",
    "you",
}

_INTENT_RULES = [
    (
        re.compile(
            r"\b(what\s+is|what\s+are|how\s+does|how\s+do|explain|define|describe|overview\s+of|introduction\s+to|basics\s+of|concept\s+of|meaning\s+of|tell\s+me\s+about)\b",
            re.I,
        ),
        INTENT_EXPLANATORY,
    ),
    (
        re.compile(
            r"\b(compare|vs\.?|versus|difference\s+between|compared\s+to|similarities|pros\s+and\s+cons|advantages\s+over|trade.?offs?)\b",
            re.I,
        ),
        INTENT_COMPARATIVE,
    ),
    (
        re.compile(
            r"\b(latest|newest|recent|state[\s._-]*of[\s._-]*the[\s._-]*art|sota|cutting[\s._-]*edge|current\s+trends?|advances?\s+in|progress\s+in|2024|2025|2026)\b",
            re.I,
        ),
        INTENT_SOTA,
    ),
    (
        re.compile(
            r"\b(derive|proof|prove|formal\s+definition|theorem|lemma|mathematical|equation\s+for|algorithm\s+for|pseudocode)\b",
            re.I,
        ),
        INTENT_TECHNICAL,
    ),
]

INTENT_TOOL_ORDER: dict[str, tuple[str, ...]] = {
    INTENT_EXPLANATORY: ("wikipedia_lookup", "tavily_search"),
    INTENT_SOTA: ("tavily_search", "wikipedia_lookup"),
    INTENT_TECHNICAL: ("calculator", "tavily_search"),
    INTENT_COMPARATIVE: ("tavily_search", "wikipedia_lookup", "calculator"),
    INTENT_DISCOVERY: ("tavily_search", "wikipedia_lookup"),
}

_FACT_TERMS: Final[tuple[str, ...]] = (
    "cash prize",
    "prize",
    "price",
    "cost",
    "date",
    "population",
    "gdp",
    "rank",
    "ranking",
    "how much",
)
_LIST_TERMS: Final[tuple[str, ...]] = ("list", "names", "examples", "top")
_COMPARISON_TERMS: Final[tuple[str, ...]] = (
    " vs ",
    "compare",
    "difference between",
    "difference",
    "versus",
)
_CALC_PREFIXES: Final[tuple[str, ...]] = ("calculate", "solve")
_ENTITY_LIKE_QUERY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .&+-]{0,40}$")
_AMBIGUOUS_HINTS = re.compile(r"\b(what|which|why|how|when|where|who|price|cost|prize|date|population|gdp|rank)\b", re.I)
_MATH_TOKEN_PATTERN = re.compile(r"\d")
_MATH_ALLOWED_PATTERN = re.compile(r"^[0-9+\-*/().,\s]+$")
_MATH_OPERATOR_PATTERN = re.compile(r"[+\-*/]")
_SINGLE_WORD_AMBIGUOUS = {"apple", "python", "java", "rust", "react", "rag"}


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text for lightweight BM25-style heuristics."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def extract_math_expression(query: str) -> str | None:
    """Extract a calculator-safe expression from a query when one is clearly present."""
    normalized = query.strip().lower()
    if not normalized:
        return None

    candidate = normalized
    for token in ["what is", "what's", "whats", "calculate", "solve", "equals", "evaluate"]:
        candidate = candidate.replace(token, " ")
    candidate = re.sub(r"[^0-9+\-*/().,\s]", " ", candidate)
    candidate = " ".join(candidate.split()).replace(",", "").strip()
    if not candidate:
        return None
    if not _MATH_TOKEN_PATTERN.search(candidate):
        return None
    if not _MATH_ALLOWED_PATTERN.fullmatch(candidate):
        return None
    if not _MATH_OPERATOR_PATTERN.search(candidate):
        return None
    return candidate


def contains_math_expression(query: str) -> bool:
    """Return whether the query clearly contains a mathematical expression."""
    normalized = query.strip().lower()
    if not normalized:
        return False

    extracted = extract_math_expression(query)
    if not extracted:
        return False

    has_calc_prefix = normalized.startswith(_CALC_PREFIXES)
    has_parentheses = "(" in extracted or ")" in extracted
    operator_count = len(_MATH_OPERATOR_PATTERN.findall(extracted))
    has_math_context = any(token in normalized for token in ["calculate", "solve", "evaluate", "compute", "what is"])
    return (has_calc_prefix or has_parentheses or operator_count >= 1) and has_math_context


def is_ambiguous_query(query: str) -> bool:
    """Return whether the query is short and underspecified enough to require clarification."""
    normalized = " ".join(query.strip().split())
    if not normalized:
        return False

    lowered = normalized.lower()
    if any(term in lowered for term in _COMPARISON_TERMS):
        return False
    if contains_math_expression(normalized):
        return False
    if _AMBIGUOUS_HINTS.search(normalized):
        return False

    word_count = len(normalized.split())
    if word_count == 1 and _ENTITY_LIKE_QUERY.fullmatch(normalized):
        return True

    if word_count == 1 and normalized.lower() in _SINGLE_WORD_AMBIGUOUS:
        return True

    return word_count <= 2 and _ENTITY_LIKE_QUERY.fullmatch(normalized) is not None


def classify_query_intent(query: str) -> str:
    """
    Classify query intent using keyword and regex rules.

    Returns one of: explanatory, sota, comparative, technical, discovery.
    """
    res = route_query(query)
    return res["intent"]


def route_query(query: str) -> dict[str, Any]:
    """
    Canonical router entry point.
    Returns: { intent, answer_type, confidence, route_source }
    """
    answer_type = classify_answer_type(query)
    query_stripped = query.strip()

    # 1. Regex fast-path
    for pattern, intent in _INTENT_RULES:
        if pattern.search(query_stripped):
            _log_classification(query_stripped, intent, 1.0, fallback=None)
            return {
                "intent": intent,
                "answer_type": answer_type,
                "confidence": 1.0,
                "route_source": "regex"
            }

    # 2. ML Classifier
    if _CLASSIFIER is not None:
        try:
            label, proba = _CLASSIFIER.predict_with_threshold(query_stripped, threshold=0.75)
            _log_classification(query_stripped, label, proba, fallback=None)
            if label is not None:
                return {
                    "intent": label,
                    "answer_type": answer_type,
                    "confidence": proba,
                    "route_source": "classifier"
                }
        except Exception:
            _log_classification(query_stripped, None, None, fallback="classifier_error")

    # 3. Safe fallback
    fallback_intent = INTENT_DISCOVERY
    _log_classification(query_stripped, fallback_intent, None, fallback="fallback")
    return {
        "intent": fallback_intent,
        "answer_type": answer_type,
        "confidence": 0.0,
        "route_source": "fallback"
    }


def classify_answer_type(query: str) -> str:
    """Classify the expected answer format for the query."""
    q = query.lower().strip()
    padded_q = f" {q} "

    # Critical edge cases: binary factual prefixes and math detection.
    if q.startswith(("is ", "was ", "are ", "does ", "did ")):
        return ANSWER_FACT

    if any(token in q for token in _FACT_TERMS):
        return ANSWER_FACT

    if any(token in q for token in _LIST_TERMS):
        return ANSWER_LIST

    if any(token in padded_q for token in _COMPARISON_TERMS):
        return ANSWER_COMPARISON

    if contains_math_expression(q):
        return ANSWER_CALCULATION

    if is_ambiguous_query(q):
        return ANSWER_AMBIGUOUS

    # Short queries are likely ambiguous
    if len(q.split()) <= 2:
        return ANSWER_AMBIGUOUS

    return ANSWER_EXPLANATION


def preferred_tools_for_intent(intent: str) -> list[str]:
    """Return the preferred tool order for an intent label."""
    return list(INTENT_TOOL_ORDER.get(intent, INTENT_TOOL_ORDER[INTENT_DISCOVERY]))
