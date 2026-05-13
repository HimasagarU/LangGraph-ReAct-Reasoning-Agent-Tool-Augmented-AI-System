from __future__ import annotations

import os
import re
import json
import time
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .intent import (
    ANSWER_AMBIGUOUS,
    ANSWER_CALCULATION,
    ANSWER_COMPARISON,
    ANSWER_EXPLANATION,
    ANSWER_FACT,
    ANSWER_LIST,
    ANSWER_MULTI,
    INTENT_COMPARATIVE,
    INTENT_DISCOVERY,
    INTENT_SOTA,
    route_query,
    extract_math_expression,
    preferred_tools_for_intent,
)
from .rewrite import rewrite_query, generate_rewrite_variants
from .memory import format_memory_context, remember_interaction
from .state import AgentState
from .tools import build_tools, calculator, score_source_quality

DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2
VALID_BUDGETS = {"shallow", "medium", "deep"}
# Deprecation shim: old depth_mode → new reasoning_budget
_DEPTH_MODE_TO_BUDGET = {"concise": "shallow", "standard": "medium", "learning_ml": "deep"}
MAX_RETRIES = 1
SYSTEM_TEMPLATE = """You are a production ReAct assistant.

Goal:
- Answer correctly.
- Put the final answer first.
- Use the smallest format that fully answers the question.
- Be precise, conservative, and easy to scan.

Rules:
- Do not invent facts, categories, dates, or sources.
- If the evidence is weak or the premise is wrong, say that clearly.
- For factual questions, answer in 1-3 short lines.
- For explanations, use a compact TL;DR-first structure.
- For comparisons, give the verdict first, then a compact table.
- For calculations, show the result first, then a minimal check.
- For ambiguous questions, ask exactly one clarification question.
- Follow the requested response format exactly.

Context:
Intent: {intent}
Answer type: {answer_type}
Reasoning budget: {reasoning_budget}
Preferred tools: {tool_order}
Rewritten query: {rewritten_query}
Execution plan: {execution_plan}
Current step: {current_step}
Memory context: {memory_context}

Response format:
{response_template}
"""
COMPARISON_TEMPLATE = """**Verdict:** [one-line recommendation or summary]

| Feature | A | B |
|---|---|---|
| [Criterion 1] | [A] | [B] |
| [Criterion 2] | [A] | [B] |

**Recommendation:** [1-2 short sentences]

Rules:
- Verdict first.
- Use only 3-4 criteria.
- Include tradeoffs, not generic filler.
"""

EXPLANATION_TEMPLATE = """**Summary:** [one-sentence answer]

**1. Intuition**
[1-2 short sentences]

**2. Breakdown**
[2-4 short sentences or bullets]

**3. Example**
[one concrete example]

**4. Takeaway**
[one short sentence]

Rules:
- Keep each section short.
- Do not force a formula if none is needed.
- Answer first, explain second.
"""

FACT_TEMPLATE = """**Answer:** [direct single-line answer]

**Sources:** [1-2 credible sources]

Rules:
- Put the direct answer first.
- If the premise is wrong, state that clearly.
- Keep it tight. Do not add extra sections.
"""

LIST_TEMPLATE = """**Answer:**
- [item 1]
- [item 2]
- [item 3]

**Sources:** [credible sources]

Rules:
- Keep items short.
- Order from most relevant to least relevant.
- Include sources for all items.
"""

CALC_TEMPLATE = """**Result:** [final numeric answer]

**Steps:** [very short verification]

Rules:
- Show the result first.
- Keep steps minimal.
- Do not use web search unless needed.
"""

AMBIGUOUS_TEMPLATE = """Clarify one thing: [one short clarification question]

Rules:
- Ask exactly one question.
- Do not answer yet.
- No extra explanation.
"""

MULTI_TEMPLATE = """### Explanation
[Your explanation following the rules]

### Calculation
[Your calculation result]

Rules:
- Keep distinct sections separated by their headers.
- Answer clearly and concisely.
"""

_URL_PATTERN = re.compile(r"https?://[^\s)]+")
_MONTH_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.I,
)


def _resolve_model_name(explicit_model_name: str | None = None) -> str:
    return explicit_model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def _resolve_temperature() -> float:
    raw_value = os.getenv("MODEL_TEMPERATURE", str(DEFAULT_TEMPERATURE))
    try:
        return float(raw_value)
    except ValueError:
        return DEFAULT_TEMPERATURE


def _resolve_model_temperature(answer_type: str) -> float:
    if answer_type == ANSWER_FACT:
        return 0.0
    return _resolve_temperature()


@lru_cache(maxsize=8)
def _load_model(model_name: str, temperature: float, api_key: str) -> ChatGroq:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to run the agent.")

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        streaming=True,
        groq_api_key=api_key,
    )

def _build_system_message(
    intent: str,
    reasoning_budget: str,
    answer_type: str,
    rewritten_query: str,
    execution_plan: str,
    current_step: str,
    memory_context: str,
    evidence_summary: str = "",
) -> SystemMessage:
    tool_order = ", ".join(preferred_tools_for_intent(intent))
    template_content = SYSTEM_TEMPLATE.format(
        intent=intent,
        answer_type=answer_type,
        reasoning_budget=reasoning_budget,
        tool_order=tool_order,
        rewritten_query=rewritten_query,
        execution_plan=execution_plan or "None",
        current_step=current_step or "answer",
        memory_context=memory_context or "None",
        response_template=_build_response_template(intent, answer_type),
    )
    if evidence_summary:
        template_content += f"\n\nEvidence pack:\n{evidence_summary}"
    return SystemMessage(content=template_content)


def _build_forced_tool_instruction(answer_type: str, tool_calls_made: list[str]) -> str:
    used_tools = {tool_name.lower() for tool_name in tool_calls_made}
    search_tools_used = any(tool in used_tools for tool in {"tavily_search", "wikipedia_lookup", "page_fetch"})

    if answer_type == ANSWER_CALCULATION:
        return "Answer directly with the numeric result only."

    if answer_type == ANSWER_FACT and not search_tools_used:
        return "You must use tavily_search, wikipedia_lookup, or page_fetch to verify this fact."

    if answer_type == ANSWER_COMPARISON:
        return (
            "You must return a verdict first, then a markdown table with 3-4 criteria, "
            "then a short recommendation. Include explicit tradeoffs."
        )

    if answer_type == ANSWER_AMBIGUOUS:
        return "Ask exactly one clarification question. Do not answer yet. Do not use tools."

    return ""


def _build_plan(intent: str, answer_type: str, query: str) -> list[str]:
    if answer_type == ANSWER_AMBIGUOUS:
        return ["clarify the request"]

    if answer_type == ANSWER_CALCULATION:
        return ["extract the expression", "calculate directly", "return the result"]

    if answer_type == ANSWER_FACT:
        return ["search for the fact", "extract the direct value", "return value source confidence"]

    if answer_type == ANSWER_COMPARISON or intent == INTENT_COMPARATIVE:
        return ["search both subjects", "compare tradeoffs", "return a structured table"]

    if intent == INTENT_SOTA:
        return ["search for current updates", "summarize the latest evidence", "return concise findings"]

    if "how" in query.lower() and answer_type == ANSWER_EXPLANATION:
        return ["search for grounding evidence", "explain the concept", "return a learning-style answer"]

    if answer_type == ANSWER_MULTI:
        return ["decompose task", "resolve explanation", "resolve calculation", "merge results"]
    
    return ["gather evidence", "answer clearly"]


def _build_planner_node(model_name: str):
    def planner_node(state: AgentState) -> dict[str, Any]:
        query = _extract_user_query(list(state.get("messages", [])))
        
        # Route the query if not already done
        if not state.get("intent") or not state.get("answer_type"):
            route_res = route_query(query)
            intent = route_res["intent"]
            answer_type = route_res["answer_type"]
            route_source = route_res["route_source"]
            classifier_confidence = route_res.get("confidence", 0.0)
            subtasks = route_res.get("subtasks", [])
        else:
            intent = str(state.get("intent", INTENT_DISCOVERY))
            answer_type = _resolve_answer_type(state)
            route_source = str(state.get("route_source", "unknown"))
            classifier_confidence = float(state.get("classifier_confidence", 0.0))
            subtasks = list(state.get("subtasks", []))

        # Compute reasoning budget: respect explicit setting, otherwise auto-compute
        mock_state = dict(state)
        mock_state["answer_type"] = answer_type
        mock_state["classifier_confidence"] = classifier_confidence
        budget = _resolve_reasoning_budget(mock_state)

        plan = _build_plan(intent, answer_type, query)
        memory_context = format_memory_context(query, budget=budget)

        # Generate rewrite variants for deep budget
        rewrite_variants: list[str] = []
        if budget == "deep":
            rewrite_variants = generate_rewrite_variants(query, answer_type, intent)

        return {
            "intent": intent,
            "answer_type": answer_type,
            "subtasks": subtasks,
            "route_source": route_source,
            "classifier_confidence": classifier_confidence,
            "classifier_label": intent if route_source == "classifier" else "",
            "plan": plan,
            "plan_index": 0,
            "reasoning_budget": budget,
            "memory_context": memory_context,
            "rewrite_variants": rewrite_variants,
            "metrics": {
                "plan_length": len(plan),
                "reasoning_budget": budget,
            },
        }

    return planner_node


def _build_response_template(intent: str, answer_type: str) -> str:
    if answer_type == ANSWER_FACT:
        return FACT_TEMPLATE

    if answer_type == ANSWER_LIST:
        return LIST_TEMPLATE

    if answer_type == ANSWER_CALCULATION:
        return CALC_TEMPLATE

    if answer_type == ANSWER_AMBIGUOUS:
        return AMBIGUOUS_TEMPLATE

    if answer_type == ANSWER_MULTI:
        return MULTI_TEMPLATE

    if answer_type == ANSWER_COMPARISON or intent == INTENT_COMPARATIVE:
        return COMPARISON_TEMPLATE

    return EXPLANATION_TEMPLATE


def _extract_user_query(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = str(message.content or "").strip()
            if content:
                return content
    return ""


def _collect_evidence(messages: list[Any]) -> str:
    evidence: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            content = str(message.content or "").strip()
            if content:
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    payload = None

                if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                    for result in payload.get("results", [])[:3]:
                        if not isinstance(result, dict):
                            continue
                        title = str(result.get("title") or "").strip()
                        url = str(result.get("url") or "").strip()
                        snippet = str(result.get("snippet") or "").strip()
                        parts = [part for part in [title, url, snippet] if part]
                        if parts:
                            evidence.append(" - ".join(parts))
                else:
                    evidence.append(content)

    if not evidence:
        return "No tool evidence available."

    return "\n\n".join(evidence[-3:])


def _resolve_reasoning_budget(state: AgentState) -> str:
    """Compute reasoning budget from state, with deprecation shim for depth_mode."""
    # Check explicit reasoning_budget first
    explicit_budget = str(state.get("reasoning_budget") or "").strip().lower()
    if explicit_budget in VALID_BUDGETS:
        return explicit_budget

    # Deprecation shim: accept old depth_mode values
    old_depth_mode = str(state.get("depth_mode", "") if "depth_mode" in state else "").strip().lower()
    if old_depth_mode in _DEPTH_MODE_TO_BUDGET:
        return _DEPTH_MODE_TO_BUDGET[old_depth_mode]

    # Auto-compute from answer_type + confidence + query complexity
    answer_type = str(state.get("answer_type") or "").strip().lower()
    classifier_confidence = float(state.get("classifier_confidence", 0.0))
    query = _extract_user_query(list(state.get("messages", [])))
    return _compute_reasoning_budget(answer_type, classifier_confidence, query)


def _compute_reasoning_budget(
    answer_type: str,
    classifier_confidence: float,
    query: str,
) -> str:
    """Map answer_type + confidence + query complexity → shallow | medium | deep."""
    # Fast paths: always shallow
    if answer_type in {ANSWER_FACT, ANSWER_CALCULATION, ANSWER_AMBIGUOUS}:
        return "shallow"

    # Multi-part: always deep
    if answer_type == ANSWER_MULTI:
        return "deep"

    # For explanation/comparison/list: check complexity signals
    word_count = len(query.split())
    is_complex_query = word_count > 8
    is_low_confidence = classifier_confidence < 0.6

    if answer_type in {ANSWER_COMPARISON, ANSWER_EXPLANATION}:
        if is_low_confidence or is_complex_query:
            return "deep"
        return "medium"

    if answer_type == ANSWER_LIST:
        return "medium"

    # Default: medium for anything unclassified
    return "medium"


def _compute_confidence(evidence: str) -> str:
    normalized = evidence.lower().strip()
    if not normalized or "no tool evidence available" in normalized:
        return "low"

    scores = [score_source_quality(url) for url in _URL_PATTERN.findall(evidence)]
    if scores and max(scores) >= 5:
        return "high"

    if any(token in normalized for token in ["wikipedia", "wiki"]):
        return "medium"

    if scores and max(scores) >= 3:
        return "medium"

    return "low"


def _plan_text(plan: list[str]) -> str:
    if not plan:
        return "None"
    return "; ".join(plan[:4])


def _current_plan_step(plan: list[str], index: int) -> str:
    if not plan:
        return "answer"
    if index < 0:
        index = 0
    if index >= len(plan):
        return "answer"
    return plan[index]


_RESULT_PREFIX_PATTERN = re.compile(r"(?is)^\s*(?:\*\*result:\*\*|result)\s*:\s*")
_FENCED_JSON_PATTERN = re.compile(r"(?is)^```(?:json)?\s*(\{.*\}|\[.*\])\s*```$")


def _parse_possible_tool_payload(raw_text: str) -> Any | None:
    text = raw_text.strip()
    if not text:
        return None

    candidates: list[str] = [text]
    stripped_result_prefix = _RESULT_PREFIX_PATTERN.sub("", text, count=1).strip()
    if stripped_result_prefix and stripped_result_prefix != text:
        candidates.append(stripped_result_prefix)

    fenced_match = _FENCED_JSON_PATTERN.match(text)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    for candidate in candidates:
        if not candidate or len(candidate) > 12000:
            continue
        if candidate[0] not in "{[":
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, (dict, list)):
            return payload

    return None


def _extract_payload_text(payload: Any) -> str | None:
    if isinstance(payload, dict):
        error_text = str(payload.get("error") or "").strip()
        if error_text:
            return error_text

        results = payload.get("results")
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    snippet = str(result.get("snippet") or "").strip()
                    if snippet:
                        return snippet
                    title = str(result.get("title") or "").strip()
                    if title:
                        return title
                elif isinstance(result, str) and result.strip():
                    return result.strip()

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                snippet = str(item.get("snippet") or "").strip()
                if snippet:
                    return snippet

    return None


def _is_raw_tool_payload_text(answer: str) -> bool:
    payload = _parse_possible_tool_payload(answer)
    if isinstance(payload, dict):
        return "results" in payload or "error" in payload
    return False


def _normalize_answer_text(answer: str, answer_type: str) -> str:
    normalized = str(answer or "").strip()
    if not normalized:
        return ""

    payload = _parse_possible_tool_payload(normalized)
    if payload is None:
        return normalized

    extracted = _extract_payload_text(payload)
    if not extracted:
        return normalized

    if answer_type == ANSWER_CALCULATION:
        return f"**Result:** {extracted}"

    if answer_type == ANSWER_MULTI:
        return normalized

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        lines: list[str] = []
        for result in payload.get("results", [])[:3]:
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or "").strip()
            snippet = str(result.get("snippet") or "").strip()
            url = str(result.get("url") or "").strip()
            item = " - ".join(part for part in [title, snippet, url] if part)
            if item:
                lines.append(item)
        if lines:
            return "**Answer:**\n" + "\n".join(f"- {line}" for line in lines)

    return extracted


def _maybe_answer_calculation(query: str) -> str | None:
    expression = extract_math_expression(query)
    if not expression:
        return None

    try:
        result = calculator.invoke({"expression": expression})
    except Exception:
        return None

    result_text = _extract_payload_text(_parse_possible_tool_payload(str(result)) or result) or str(result).strip()
    if not result_text:
        return None
    return f"**Result:** {result_text}"


def _resolve_answer_type(state: AgentState) -> str:
    answer_type = str(state.get("answer_type") or "").strip().lower()
    valid = {
        ANSWER_FACT,
        ANSWER_LIST,
        ANSWER_EXPLANATION,
        ANSWER_COMPARISON,
        ANSWER_CALCULATION,
        ANSWER_AMBIGUOUS,
        ANSWER_MULTI,
    }
    if answer_type in valid:
        return answer_type

    query = _extract_user_query(list(state.get("messages", [])))
    if query:
        return route_query(query)["answer_type"]

    return ANSWER_EXPLANATION


def _extract_tool_names(ai_message: AIMessage) -> list[str]:
    tool_calls = getattr(ai_message, "tool_calls", []) or []
    tool_names: list[str] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_name = str(tool_call.get("name", ""))
        else:
            tool_name = str(getattr(tool_call, "name", ""))
        if tool_name:
            tool_names.append(tool_name)
    return tool_names


def _answer_quality_issues(
    answer: str,
    answer_type: str,
    intent: str,
    reasoning_budget: str = "medium",
    tool_calls_made: list[str] | None = None,
) -> list[str]:
    normalized_answer = answer.lower()
    issues: list[str] = []

    def has_table() -> bool:
        return any("|" in line for line in answer.splitlines())

    def has_tradeoff_language() -> bool:
        return any(
            token in normalized_answer
            for token in ["tradeoff", "trade-off", "best for", "better when", "use case", "works well", "less suited"]
        )

    if _is_raw_tool_payload_text(answer):
        issues.append("raw tool payload leaked into final answer")

    # ── Format checks (all budgets) ──────────────────────────────────────────
    if answer_type == ANSWER_COMPARISON or intent == INTENT_COMPARATIVE:
        if not has_table():
            issues.append("missing comparison table")
        if not has_tradeoff_language():
            issues.append("missing tradeoff guidance")
        if "depends on context" in normalized_answer:
            issues.append("bare fallback phrase")
    elif answer_type == ANSWER_FACT:
        if not answer.strip():
            issues.append("empty answer")
        if "not specified" in normalized_answer:
            issues.append("likely missing retrieval")
    elif answer_type == ANSWER_LIST:
        if not re.search(r"(?m)^(?:[-*]\s+|\d+\.)", answer):
            issues.append("missing bullet list")
    elif answer_type == ANSWER_CALCULATION:
        if not re.search(r"(?im)^\*\*result:\*\*\s*", answer) and not re.search(r"\d", answer):
            issues.append("missing numeric result")
    elif answer_type == ANSWER_AMBIGUOUS:
        stripped = answer.strip()
        if not stripped.endswith("?") or stripped.count("?") != 1:
            issues.append("must ask exactly one clarification question")
        if re.search(r"(?im)^\*\*(?:answer|sources|result):\*\*", answer):
            issues.append("should not answer before clarification")
    elif answer_type == ANSWER_MULTI:
        if "### Explanation" not in answer and "### Calculation" not in answer:
            issues.append("missing multi-part headers")
    else:
        has_summary = re.search(r"(?im)\*\*summary:\*\*", answer)
        has_intuition = re.search(r"(?im)\*\*1\.\s*intuition\*\*|\*\*intuition\*\*", answer)
        has_breakdown = re.search(r"(?im)\*\*2\.\s*breakdown\*\*|\*\*breakdown\*\*", answer)
        has_any_structure = has_summary or has_intuition or has_breakdown
        if not has_any_structure and len(answer.strip()) < 40:
            issues.append("answer too short")

    # ── Content-aware critic checks (deep budget only) ───────────────────────
    if reasoning_budget == "deep":
        issues.extend(_critic_content_checks(answer, answer_type, intent))

    return issues


def _critic_content_checks(answer: str, answer_type: str, intent: str) -> list[str]:
    """Heuristic content checks — no LLM calls. Only runs for deep budget."""
    normalized = answer.lower()
    issues: list[str] = []

    # 1) Overclaim detection: absolute language without hedging
    _OVERCLAIM_TOKENS = ["always ", "never ", " all ", " none ", "every single", "impossible to"]
    _HEDGE_TOKENS = ["generally", "typically", "in most cases", "often", "usually", "tends to"]
    has_overclaim = any(tok in normalized for tok in _OVERCLAIM_TOKENS)
    has_hedge = any(tok in normalized for tok in _HEDGE_TOKENS)
    if has_overclaim and not has_hedge:
        issues.append("overclaim_detected: uses absolute language without hedging")

    # 2) Common ML/AI taxonomy confusion
    _CONFUSION_PAIRS = [
        ("agentic ai", "ai agent"),
        ("machine learning", "deep learning"),
        ("supervised", "unsupervised"),
        ("classification", "regression"),
    ]
    for term_a, term_b in _CONFUSION_PAIRS:
        if term_a in normalized and term_b in normalized:
            if f"{term_a} is {term_b}" in normalized or f"{term_b} is {term_a}" in normalized:
                issues.append(f"taxonomy_confusion: conflates '{term_a}' with '{term_b}'")

    # 3) Empty substance check: answer has length but no real content
    if len(answer.strip()) > 100:
        filler_count = sum(
            1 for phrase in ["it depends", "there are many", "it varies", "in general"]
            if phrase in normalized
        )
        if filler_count >= 2:
            issues.append("low_substance: answer relies on filler phrases")

    return issues


def _restore_collapsed_formatting(answer: str, answer_type: str) -> str:
    """Restore newlines in collapsed formatting (e.g., Summary: ... 1. Intuition ... 2. Breakdown ...)."""
    if answer_type not in {ANSWER_EXPLANATION, ANSWER_FACT, ANSWER_LIST}:
        return answer
    
    # If already properly formatted with newlines, return as-is
    if answer.count('\n') >= 3:
        return answer
    
    # Pattern for collapsed explanation: **Summary:** ... **1. Intuition** ... **2. Breakdown** ...
    # Replace ** followed by section header with newline + header
    result = re.sub(r'(\S)\s+\*\*(\d+\.\s+(?:Intuition|Breakdown|Example|Takeaway))\*\*', 
                   r'\1\n\n**\2**', answer, flags=re.I)
    
    # Also handle Summary header followed by content
    result = re.sub(r'(\*\*Summary:\*\*\s+\S.*?)(\*\*\d+\.)', 
                   r'\1\n\n\2', result, flags=re.I)
    
    # Add newline after section headers if content follows immediately
    result = re.sub(r'(\*\*\d+\.\s+(?:Intuition|Breakdown|Example|Takeaway)\*\*)\s+([A-Z])', 
                   r'\1\n\2', result, flags=re.I)
    result = re.sub(r'(\*\*Summary:\*\*)\s+([A-Z])', 
                   r'\1\n\2', result, flags=re.I)
    
    # Handle Answer header for lists/facts
    result = re.sub(r'(\*\*Answer:\*\*)\s+([-*])', 
                   r'\1\n\2', result, flags=re.I)
    
    # Handle Sources header
    result = re.sub(r'(\S)\s+(\*\*Sources:\*\*)', 
                   r'\1\n\n\2', result, flags=re.I)
    
    return result


def _safe_fallback_answer(answer_type: str) -> str:
    if answer_type == ANSWER_FACT:
        return "**Answer:** Information not found from the available sources.\n\n**Sources:** Tool results"
    if answer_type == ANSWER_COMPARISON:
        return (
            "**Verdict:** Unable to verify a reliable comparison from available sources.\n\n"
            "**Recommendation:** Review more authoritative sources and retry."
        )
    if answer_type == ANSWER_LIST:
        return "**Answer:**\n- Information not found from the available sources."
    if answer_type == ANSWER_CALCULATION:
        return "**Result:** Calculation could not be verified.\n\n**Steps:** Unable to validate with available evidence."
    if answer_type == ANSWER_AMBIGUOUS:
        return "Clarify one thing: Could you clarify what you mean?"
    return "**TL;DR:** I could not produce a reliable answer from the available sources."


def _time_budget_exceeded(state: AgentState) -> bool:
    started_at = state.get("started_at")
    max_wall_time_s = state.get("max_wall_time_s")
    if started_at is None or max_wall_time_s is None:
        return False
    try:
        return (time.perf_counter() - float(started_at)) > float(max_wall_time_s)
    except (TypeError, ValueError):
        return False


def _build_timeout_node():
    def timeout_node(state: AgentState) -> dict[str, Any]:
        answer_type = _resolve_answer_type(state)
        iteration_count = int(state.get("iteration_count", 0))
        max_iterations = int(state.get("max_iterations", 0))
        reason = "time budget exceeded" if _time_budget_exceeded(state) else "max iterations reached"
        safe_answer = _safe_fallback_answer(answer_type)
        return {
            "messages": [AIMessage(content=safe_answer)],
            "final_answer": safe_answer,
            "final_answer_reviewed": True,
            "needs_retry": False,
            "validation_errors": [reason],
            "answer_format_ok": False,
            "metrics": {
                "plan_length": len(state.get("plan", []) or []),
                "steps": iteration_count,
                "reasoning_budget": _resolve_reasoning_budget(state),
                "stop_reason": reason,
                "max_iterations": max_iterations,
            },
        }

    return timeout_node


def _build_agent_node(model_name: str, temperature: float):
    tools = build_tools()

    def agent_node(state: AgentState) -> dict[str, Any]:
        intent = state.get("intent", INTENT_DISCOVERY)
        budget = _resolve_reasoning_budget(state)
        answer_type = _resolve_answer_type(state)
        history = list(state.get("messages", []))
        user_query = _extract_user_query(history)
        evidence = _collect_evidence(history)
        confidence = _compute_confidence(evidence)
        rewritten_query = rewrite_query(user_query, answer_type, intent, budget=budget) if user_query else ""
        plan = list(state.get("plan", []))
        plan_index = int(state.get("plan_index", 0))
        current_step = _current_plan_step(plan, plan_index)
        memory_context = str(state.get("memory_context") or "").strip() or format_memory_context(user_query, budget=budget)
        tool_calls_made = list(state.get("tool_calls_made", []))
        forced_instruction = _build_forced_tool_instruction(answer_type, tool_calls_made)
        effective_temperature = _resolve_model_temperature(answer_type)

        # Build evidence summary from evidence_pack if available
        evidence_pack = state.get("evidence_pack") or {}
        evidence_summary = evidence_pack.get("summary", "") if evidence_pack else ""

        if answer_type == ANSWER_AMBIGUOUS and user_query:
            clarification = f'Could you clarify what you mean by "{user_query.strip()}"?'
            return {
                "messages": [AIMessage(content=clarification)],
                "final_answer": clarification,
                "final_answer_reviewed": False,
                "reasoning_budget": budget,
                "answer_type": answer_type,
                "plan": plan,
                "plan_index": plan_index,
                "rewritten_query": rewritten_query,
                "memory_context": memory_context,
                "confidence": "low",
                "needs_retry": False,
                "validation_errors": [],
                "answer_format_ok": False,
                "metrics": {"plan_length": len(plan), "steps": int(state.get("iteration_count", 0)), "reasoning_budget": budget},
            }

        if answer_type == ANSWER_CALCULATION and user_query:
            calculation_answer = _maybe_answer_calculation(user_query)
            if calculation_answer:
                return {
                    "messages": [AIMessage(content=calculation_answer)],
                    "final_answer": calculation_answer,
                    "final_answer_reviewed": False,
                    "reasoning_budget": budget,
                    "answer_type": answer_type,
                    "plan": plan,
                    "plan_index": plan_index,
                    "rewritten_query": rewritten_query,
                    "memory_context": memory_context,
                    "confidence": "high",
                    "needs_retry": False,
                    "validation_errors": [],
                    "answer_format_ok": False,
                    "metrics": {"plan_length": len(plan), "steps": int(state.get("iteration_count", 0)), "reasoning_budget": budget},
                }
        model = _load_model(model_name, effective_temperature, os.getenv("GROQ_API_KEY", ""))
        model_with_tools = model.bind_tools(tools)
        
        if answer_type == "multi":
            parts = []
            subtasks = state.get("subtasks", [])
            
            if "explanation" in subtasks:
                sys_expl = _build_system_message("explanation", budget, ANSWER_EXPLANATION, rewritten_query, _plan_text(plan), current_step, memory_context, evidence_summary)
                if forced_instruction:
                    sys_expl = SystemMessage(content=f"{sys_expl.content}\n\n{forced_instruction}")
                isolate_msg = HumanMessage(content="CRITICAL INSTRUCTION: You are executing the 'explanation' portion of a multi-part query. ONLY explain the concept. Do NOT perform the calculation or answer other parts.")
                resp_expl = model_with_tools.invoke([sys_expl, *history, isolate_msg])
                parts.append(f"### Explanation\n{str(resp_expl.content or '').strip()}")
                
            if "calculation" in subtasks:
                calc_ans = _maybe_answer_calculation(user_query)
                if calc_ans:
                    parts.append(f"### Calculation\n{calc_ans}")
                else:
                    sys_calc = _build_system_message("calculation", budget, ANSWER_CALCULATION, rewritten_query, _plan_text(plan), current_step, memory_context)
                    isolate_msg = HumanMessage(content="CRITICAL INSTRUCTION: You are executing the 'calculation' portion of a multi-part query. ONLY perform the calculation. Do NOT explain other concepts.")
                    resp_calc = model_with_tools.invoke([sys_calc, *history, isolate_msg])
                    parts.append(f"### Calculation\n{str(resp_calc.content or '').strip()}")
            
            final_answer = "\n\n".join(parts)
            return {
                "messages": [AIMessage(content=final_answer)],
                "final_answer": final_answer,
                "final_answer_reviewed": False,
                "reasoning_budget": budget,
                "answer_type": answer_type,
                "plan": plan,
                "plan_index": min(plan_index + 1, len(plan)),
                "rewritten_query": rewritten_query,
                "memory_context": memory_context,
                "confidence": confidence,
                "needs_retry": False,
                "validation_errors": [],
                "answer_format_ok": False,
                "iteration_count": int(state.get("iteration_count", 0)) + 1,
                "metrics": {"plan_length": len(plan), "steps": int(state.get("iteration_count", 0)) + 1, "reasoning_budget": budget},
            }

        system_message = _build_system_message(
            intent,
            budget,
            answer_type,
            rewritten_query,
            _plan_text(plan),
            current_step,
            memory_context,
            evidence_summary,
        )
        if forced_instruction:
            system_message = SystemMessage(content=f"{system_message.content}\n\n{forced_instruction}")

        response = model_with_tools.invoke([system_message, *history])

        updates: dict[str, Any] = {
            "messages": [response],
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
            "reasoning_budget": budget,
            "answer_type": answer_type,
            "plan": plan,
            "plan_index": min(plan_index + 1, len(plan)),
            "rewritten_query": rewritten_query,
            "memory_context": memory_context,
            "final_answer": "",
            "final_answer_reviewed": False,
            "needs_retry": False,
            "confidence": confidence,
            "validation_errors": [],
            "answer_format_ok": False,
            "metrics": {"plan_length": len(plan), "steps": int(state.get("iteration_count", 0)) + 1, "reasoning_budget": budget},
        }

        tool_names = _extract_tool_names(response)
        if tool_names:
            updates["tool_calls_made"] = [*state.get("tool_calls_made", []), *tool_names]
        else:
            content = str(response.content or "").strip()
            if content:
                updates["final_answer"] = _normalize_answer_text(content, answer_type)
                updates["final_answer_reviewed"] = False

        return updates

    return agent_node, tools


def _build_evidence_pack_node():
    """Pure Python node: aggregates tool outputs into a structured evidence pack.

    No LLM calls — zero extra token cost. Runs between agent and review.
    For shallow budget, passes through without building a pack.
    """

    def evidence_pack_node(state: AgentState) -> dict[str, Any]:
        budget = _resolve_reasoning_budget(state)
        messages = list(state.get("messages", []))
        user_query = _extract_user_query(messages)

        # Shallow budget: skip evidence aggregation entirely
        if budget == "shallow":
            return {"evidence_pack": {}, "critic_issues": []}

        # Collect all tool evidence
        snippets: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            content = str(message.content or "").strip()
            if not content:
                continue

            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                for result in payload.get("results", [])[:5]:
                    if not isinstance(result, dict):
                        continue
                    url = str(result.get("url") or "").strip()
                    title = str(result.get("title") or "").strip()
                    snippet = str(result.get("snippet") or "").strip()
                    score = int(result.get("score", 0)) or score_source_quality(url, snippet, title)
                    if url or snippet:
                        snippets.append({"title": title, "url": url, "snippet": snippet, "score": score})
            else:
                # Raw text tool output (e.g. wikipedia)
                snippets.append({"title": "", "url": "", "snippet": content[:300], "score": 2})

        # Sort by quality score descending
        snippets.sort(key=lambda s: s.get("score", 0), reverse=True)

        # Build contradiction flags (simple heuristic: check for negation patterns)
        contradiction_flags: list[str] = []
        if len(snippets) >= 2:
            texts = [s.get("snippet", "").lower() for s in snippets[:4]]
            _NEGATION_PAIRS = [
                ("is not", "is"),
                ("does not", "does"),
                ("cannot", "can"),
                ("no longer", "still"),
            ]
            for neg, pos in _NEGATION_PAIRS:
                has_neg = any(neg in t for t in texts)
                has_pos = any(pos in t and neg not in t for t in texts)
                if has_neg and has_pos:
                    contradiction_flags.append(f"possible contradiction: '{neg}' vs '{pos}' in sources")

        # Build summary string for injection into system prompt
        summary_lines = [f"Query: {user_query}"]
        rewrite_variants = list(state.get("rewrite_variants", []))
        if rewrite_variants:
            summary_lines.append(f"Rewrite variants: {'; '.join(rewrite_variants[:4])}")
        summary_lines.append(f"Evidence ({len(snippets)} sources, sorted by quality):")
        for i, s in enumerate(snippets[:5], 1):
            title = s.get("title", "")
            snippet = s.get("snippet", "")[:200]
            score = s.get("score", 0)
            summary_lines.append(f"  [{i}] (score={score}) {title}: {snippet}")
        if contradiction_flags:
            summary_lines.append(f"Contradictions: {'; '.join(contradiction_flags)}")

        evidence_pack = {
            "query": user_query,
            "rewrite_variants": rewrite_variants,
            "snippets": snippets[:5],
            "contradiction_flags": contradiction_flags,
            "source_count": len(snippets),
            "summary": "\n".join(summary_lines),
        }

        # Run critic checks for deep budget
        critic_issues: list[str] = []
        if budget == "deep":
            draft = str(state.get("final_answer") or "").strip()
            if draft:
                answer_type = _resolve_answer_type(state)
                intent = str(state.get("intent", INTENT_DISCOVERY))
                critic_issues = _critic_content_checks(draft, answer_type, intent)

        return {
            "evidence_pack": evidence_pack,
            "critic_issues": critic_issues,
        }

    return evidence_pack_node


def _build_review_node(model_name: str, temperature: float):
    def review_node(state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        draft_answer = str(state.get("final_answer") or "").strip()
        if not draft_answer:
            draft_answer = _best_available_answer_from_messages(messages)
        draft_answer = _normalize_answer_text(draft_answer, _resolve_answer_type(state))

        if not draft_answer:
            return {"final_answer_reviewed": True}

        query = _extract_user_query(messages)
        intent = str(state.get("intent", INTENT_DISCOVERY))
        answer_type = _resolve_answer_type(state)
        budget = _resolve_reasoning_budget(state)
        evidence = _collect_evidence(messages)
        issues = _answer_quality_issues(
            draft_answer,
            answer_type,
            intent,
            reasoning_budget=budget,
            tool_calls_made=list(state.get("tool_calls_made", [])),
        )

        # Merge critic issues from evidence_pack phase
        critic_issues = list(state.get("critic_issues", []))
        if critic_issues:
            issues.extend(critic_issues)

        if answer_type in {ANSWER_AMBIGUOUS, ANSWER_CALCULATION, ANSWER_MULTI, ANSWER_EXPLANATION}:
            draft_answer = _normalize_answer_text(draft_answer, answer_type)
            return {
                "messages": [AIMessage(content=draft_answer)],
                "final_answer": draft_answer,
                "final_answer_reviewed": True,
                "reasoning_budget": budget,
                "validation_errors": issues,
                "answer_format_ok": not issues,
            }

        effective_temperature = _resolve_model_temperature(answer_type)
        model = _load_model(model_name, effective_temperature, os.getenv("GROQ_API_KEY", ""))

        prompt = [
            SystemMessage(
                content="""You are an extremely strict final-answer formatter and verifier.

Required output structure (answer-first):
{response_template}

Question: {query}
Evidence: {evidence}

Previous issues: {issues}

Instructions:
- Return the direct answer first, then a one-line justification, then caveats if needed, then sources/confidence when applicable.
- Fix ALL format issues.
- Do not add information not present in the evidence.
- Be precise with names, dates, and categories.
- CRITICAL: Preserve all newlines and section headers exactly.
- Each section (Summary, Intuition, Breakdown, Example, Takeaway) MUST be on a separate line with its header.
- Use exactly one blank line between sections.
- Return nothing except the correctly formatted answer.
""".format(
                    response_template=_build_response_template(intent, answer_type),
                    issues="\n".join(f"- {issue}" for issue in issues) if issues else "- None",
                    query=query,
                    evidence=evidence,
                )
            ),
            HumanMessage(content="Rewrite the draft and return only the final answer in the required format. Preserve all newlines and structure."),
        ]

        revised_answer = draft_answer
        for _ in range(2):
            response = model.invoke(prompt)
            candidate_answer = str(response.content or "").strip()
            if candidate_answer:
                candidate_answer = _restore_collapsed_formatting(candidate_answer, answer_type)
                candidate_answer = _normalize_answer_text(candidate_answer, answer_type)
                revised_answer = candidate_answer
            issues = _answer_quality_issues(
                revised_answer,
                answer_type,
                intent,
                reasoning_budget=budget,
                tool_calls_made=list(state.get("tool_calls_made", [])),
            )
            if not issues:
                break
            prompt[0] = SystemMessage(
                content="""You are a final answer reviewer.

Required output structure (answer-first):
{response_template}

Review rules:
- Ensure the direct answer appears first, then a concise justification, then caveats, then sources/confidence when applicable.
- Check whether any statement is misleading, overly general, too certain, too dense, or unsupported.
- Do not add new facts.
- Do not add new sources, URLs, numbers, or named entities that are not already in the draft or evidence.
- Do not use a bare fallback like "depends on context".
- If a comparison is not fully supported, explain the tradeoffs inside the table or use cases.
- Preserve the required structure.
- Fix the issues listed below.

Issues to fix:
{issues}

Question:
{query}

Intent:
{intent}

Answer type:
{answer_type}

Draft answer:
{draft_answer}

Evidence:
{evidence}
""".format(
                    response_template=_build_response_template(intent, answer_type),
                    issues="\n".join(f"- {issue}" for issue in issues),
                    query=query,
                    intent=intent,
                    answer_type=answer_type,
                    draft_answer=revised_answer,
                    evidence=evidence,
                )
            )

        revised_answer = _normalize_answer_text(_restore_collapsed_formatting(revised_answer, answer_type), answer_type)
        
        return {
            "messages": [AIMessage(content=revised_answer)],
            "final_answer": revised_answer,
            "final_answer_reviewed": True,
            "reasoning_budget": budget,
            "validation_errors": issues,
            "answer_format_ok": not issues,
        }

    return review_node


def _build_validate_node():
    def validate_node(state: AgentState) -> dict[str, Any]:
        answer_type = _resolve_answer_type(state)
        answer = _normalize_answer_text(str(state.get("final_answer") or "").strip(), answer_type)
        messages = list(state.get("messages", []))
        intent = str(state.get("intent", INTENT_DISCOVERY))
        budget = _resolve_reasoning_budget(state)

        # Budget-aware max retries: shallow=0, medium=1, deep=2
        max_retries = {"shallow": 0, "medium": 1, "deep": 2}.get(budget, MAX_RETRIES)

        issues = _answer_quality_issues(
            answer,
            answer_type,
            intent,
            reasoning_budget=budget,
            tool_calls_made=list(state.get("tool_calls_made", [])),
        )

        if not answer:
            issues.append("empty answer")

        retry_count = int(state.get("retry_count", 0))
        if issues and retry_count < max_retries:
            rewritten_query = str(state.get("rewritten_query") or "").strip()
            issues_text = "\n- ".join(issues)
            retry_count += 1
            return {
                "messages": [
                    SystemMessage(
                        content=(
                            "Validation failed. Retry with the required structure and use tools if needed.\n"
                            f"Issues:\n- {issues_text}\n"
                            f"Rewritten query for tools: {rewritten_query}\n"
                            "Return only the final answer."
                        )
                    )
                ],
                "final_answer": "",
                "final_answer_reviewed": False,
                "retry_count": retry_count,
                "needs_retry": True,
                "validation_errors": issues,
                "answer_format_ok": False,
                "critic_issues": issues,
            }

        if issues:
            safe_answer = _safe_fallback_answer(answer_type)
            return {
                "messages": [AIMessage(content=safe_answer)],
                "final_answer": safe_answer,
                "final_answer_reviewed": True,
                "retry_count": retry_count,
                "needs_retry": False,
                "validation_errors": issues,
                "answer_format_ok": False,
            }

        if not issues and answer:
            query = _extract_user_query(messages)
            remember_interaction(
                query,
                answer,
                {
                    "intent": intent,
                    "answer_type": answer_type,
                    "confidence": state.get("confidence"),
                    "tools_used": list(state.get("tool_calls_made", [])),
                    "reasoning_budget": budget,
                },
                budget=budget,
            )

        return {
            "needs_retry": False,
            "final_answer": answer,
            "validation_errors": issues,
            "answer_format_ok": not issues,
        }

    return validate_node


def _best_available_answer_from_messages(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = str(message.content or "").strip()
            if content:
                return _normalize_answer_text(content, _resolve_answer_type({"messages": messages}))

    observations: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            content = str(message.content or "").strip()
            if content:
                observations.append(content)

    if observations:
        return _normalize_answer_text(observations[-1], _resolve_answer_type({"messages": messages}))

    return ""


def _should_continue(state: AgentState) -> str:
    max_iterations = int(state.get("max_iterations", 5))
    iteration_count = int(state.get("iteration_count", 0))
    final_answer = str(state.get("final_answer") or "").strip()
    final_answer_reviewed = bool(state.get("final_answer_reviewed", False))

    if _time_budget_exceeded(state):
        if final_answer and not final_answer_reviewed:
            return "evidence_pack"
        return "timeout"

    if iteration_count >= max_iterations:
        if final_answer and not final_answer_reviewed:
            return "evidence_pack"
        return "timeout"

    if (
        str(state.get("confidence") or "").lower() == "high"
        and final_answer_reviewed
        and not bool(state.get("needs_retry", False))
    ):
        return END

    messages = list(state.get("messages", []))
    if not messages:
        return END

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"

    if isinstance(last_message, AIMessage) and final_answer:
        if not final_answer_reviewed:
            return "evidence_pack"
        if bool(state.get("needs_retry", False)):
            return "agent"

    return END


def _should_retry_after_validation(state: AgentState) -> str:
    if bool(state.get("needs_retry", False)):
        return "agent"

    return END


@lru_cache(maxsize=8)
def build_graph(model_name: str | None = None, temperature: float | None = None):
    resolved_model_name = _resolve_model_name(model_name)
    resolved_temperature = _resolve_temperature() if temperature is None else temperature
    planner_node = _build_planner_node(resolved_model_name)
    agent_node, tools = _build_agent_node(resolved_model_name, resolved_temperature)
    evidence_pack_node = _build_evidence_pack_node()
    review_node = _build_review_node(resolved_model_name, resolved_temperature)
    validate_node = _build_validate_node()
    timeout_node = _build_timeout_node()
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("evidence_pack_builder", evidence_pack_node)
    workflow.add_node("review", review_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("timeout", timeout_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "evidence_pack": "evidence_pack_builder", "timeout": "timeout", END: END},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("evidence_pack_builder", "review")
    workflow.add_edge("review", "validate")
    workflow.add_conditional_edges("validate", _should_retry_after_validation, {"agent": "agent", END: END})
    workflow.add_edge("timeout", END)
    return workflow.compile()
