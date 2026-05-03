from __future__ import annotations

import os
import re
import json
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
from .rewrite import rewrite_query
from .memory import format_memory_context, remember_interaction
from .state import AgentState
from .tools import build_tools, calculator, score_source_quality

DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_DEPTH_MODE = "learning_ml"
VALID_DEPTH_MODES = {"learning_ml", "standard", "concise"}
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
Depth mode: {depth_mode}
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

Rules:
- Keep items short.
- Order from most relevant to least relevant.
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

def _build_system_message_for_depth(
    intent: str,
    depth_mode: str,
    answer_type: str,
    rewritten_query: str,
    execution_plan: str,
    current_step: str,
    memory_context: str,
) -> SystemMessage:
    tool_order = ", ".join(preferred_tools_for_intent(intent))
    return SystemMessage(
        content=SYSTEM_TEMPLATE.format(
            intent=intent,
            answer_type=answer_type,
            depth_mode=depth_mode,
            tool_order=tool_order,
            rewritten_query=rewritten_query,
            execution_plan=execution_plan or "None",
            current_step=current_step or "answer",
            memory_context=memory_context or "None",
            response_template=_build_response_template(intent, depth_mode, answer_type),
        )
    )


def _build_forced_tool_instruction(answer_type: str, tool_calls_made: list[str]) -> str:
    used_tools = {tool_name.lower() for tool_name in tool_calls_made}
    search_tools_used = any(tool in used_tools for tool in {"tavily_search", "wikipedia_lookup"})

    if answer_type == ANSWER_CALCULATION:
        return ""

    if answer_type == ANSWER_FACT:
        if not search_tools_used:
            return "You must use tavily_search or wikipedia_lookup to verify this fact."

    if answer_type == ANSWER_AMBIGUOUS:
        return "Ask one clarification question only. Do not search."

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

        plan = _build_plan(intent, answer_type, query)
        memory_context = format_memory_context(query, limit=2)

        return {
            "intent": intent,
            "answer_type": answer_type,
            "subtasks": subtasks,
            "route_source": route_source,
            "classifier_confidence": classifier_confidence,
            "classifier_label": intent if route_source == "classifier" else "",
            "plan": plan,
            "plan_index": 0,
            "memory_context": memory_context,
            "metrics": {
                "plan_length": len(plan),
            },
        }

    return planner_node


def _build_response_template(intent: str, depth_mode: str, answer_type: str) -> str:
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
    if depth_mode == "concise":
        return EXPLANATION_TEMPLATE + "\n- Keep each section brief, but still include an example and a step breakdown."

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


def _resolve_depth_mode(state: AgentState) -> str:
    explicit_depth_mode = str(state.get("depth_mode") or "").strip().lower()
    if explicit_depth_mode in VALID_DEPTH_MODES:
        return explicit_depth_mode

    query = _extract_user_query(list(state.get("messages", []))).lower()
    if any(keyword in query for keyword in ["concise", "brief", "short answer", "short version"]):
        return "concise"

    if any(
        keyword in query
        for keyword in [
            "explain like i'm learning ml",
            "explain like i am learning ml",
            "give intuition + math + example",
            "intuition + math + example",
            "intuition, math, and example",
        ]
    ):
        return "learning_ml"

    return DEFAULT_DEPTH_MODE


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


def _maybe_answer_calculation(query: str) -> str | None:
    expression = extract_math_expression(query)
    if not expression:
        return None

    try:
        result = calculator.invoke({"expression": expression})
    except Exception:
        return None

    result_text = str(result).strip()
    if not result_text:
        return None
    return f"Result: {result_text}"


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
    depth_mode: str,
    tool_calls_made: list[str] | None = None,
) -> list[str]:
    normalized_answer = answer.lower()
    issues: list[str] = []
    used_tools = {tool_name.lower() for tool_name in (tool_calls_made or [])}
    search_tools_used = any(tool in used_tools for tool in {"tavily_search", "wikipedia_lookup"})

    def has_text(text: str) -> bool:
        return text.lower() in normalized_answer

    def has_table() -> bool:
        return any("|" in line for line in answer.splitlines())

    def has_tradeoff_language() -> bool:
        return any(
            token in normalized_answer
            for token in ["tradeoff", "trade-off", "best for", "better when", "use case", "works well", "less suited"]
        )

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
        if not re.search(r"(?im)^\*\*summary:\*\*\s*", answer):
            issues.append("missing Summary")
        if len(answer.strip()) < 40:
            issues.append("answer too short")
        if depth_mode == "learning_ml":
            if not any(token in normalized_answer for token in ["intuition", "example", "formula", "breakdown"]):
                issues.append("missing learning-style structure")

    return issues


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


def _build_agent_node(model_name: str, temperature: float):
    tools = build_tools()

    def agent_node(state: AgentState) -> dict[str, Any]:
        intent = state.get("intent", INTENT_DISCOVERY)
        depth_mode = _resolve_depth_mode(state)
        answer_type = _resolve_answer_type(state)
        history = list(state.get("messages", []))
        user_query = _extract_user_query(history)
        evidence = _collect_evidence(history)
        confidence = _compute_confidence(evidence)
        rewritten_query = rewrite_query(user_query, answer_type, intent) if user_query else ""
        plan = list(state.get("plan", []))
        plan_index = int(state.get("plan_index", 0))
        current_step = _current_plan_step(plan, plan_index)
        memory_context = str(state.get("memory_context") or "").strip() or format_memory_context(user_query, limit=2)
        tool_calls_made = list(state.get("tool_calls_made", []))
        forced_instruction = _build_forced_tool_instruction(answer_type, tool_calls_made)
        effective_temperature = _resolve_model_temperature(answer_type)

        if answer_type == ANSWER_AMBIGUOUS and user_query:
            clarification = f'Could you clarify what you mean by "{user_query.strip()}"?'
            return {
                "messages": [AIMessage(content=clarification)],
                "final_answer": clarification,
                "final_answer_reviewed": False,
                "depth_mode": depth_mode,
                "answer_type": answer_type,
                "plan": plan,
                "plan_index": plan_index,
                "rewritten_query": rewritten_query,
                "memory_context": memory_context,
                "confidence": "low",
                "needs_retry": False,
                "validation_errors": [],
                "answer_format_ok": False,
                "metrics": {"plan_length": len(plan), "steps": int(state.get("iteration_count", 0))},
            }

        if answer_type == ANSWER_CALCULATION and user_query:
            calculation_answer = _maybe_answer_calculation(user_query)
            if calculation_answer:
                return {
                    "messages": [AIMessage(content=calculation_answer)],
                    "final_answer": calculation_answer,
                    "final_answer_reviewed": False,
                    "depth_mode": depth_mode,
                    "answer_type": answer_type,
                    "plan": plan,
                    "plan_index": plan_index,
                    "rewritten_query": rewritten_query,
                    "memory_context": memory_context,
                    "confidence": "high",
                    "needs_retry": False,
                    "validation_errors": [],
                    "answer_format_ok": False,
                    "metrics": {"plan_length": len(plan), "steps": int(state.get("iteration_count", 0))},
                }
        model = _load_model(model_name, effective_temperature, os.getenv("GROQ_API_KEY", ""))
        model_with_tools = model.bind_tools(tools)
        
        if answer_type == "multi":
            parts = []
            subtasks = state.get("subtasks", [])
            
            if "explanation" in subtasks:
                sys_expl = _build_system_message_for_depth("explanation", depth_mode, ANSWER_EXPLANATION, rewritten_query, _plan_text(plan), current_step, memory_context)
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
                    sys_calc = _build_system_message_for_depth("calculation", depth_mode, ANSWER_CALCULATION, rewritten_query, _plan_text(plan), current_step, memory_context)
                    isolate_msg = HumanMessage(content="CRITICAL INSTRUCTION: You are executing the 'calculation' portion of a multi-part query. ONLY perform the calculation. Do NOT explain other concepts.")
                    resp_calc = model_with_tools.invoke([sys_calc, *history, isolate_msg])
                    parts.append(f"### Calculation\n{str(resp_calc.content or '').strip()}")
            
            final_answer = "\n\n".join(parts)
            return {
                "messages": [AIMessage(content=final_answer)],
                "final_answer": final_answer,
                "final_answer_reviewed": False,
                "depth_mode": depth_mode,
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
                "metrics": {"plan_length": len(plan), "steps": int(state.get("iteration_count", 0)) + 1},
            }

        system_message = _build_system_message_for_depth(
            intent,
            depth_mode,
            answer_type,
            rewritten_query,
            _plan_text(plan),
            current_step,
            memory_context,
        )
        if forced_instruction:
            system_message = SystemMessage(content=f"{system_message.content}\n\n{forced_instruction}")

        response = model_with_tools.invoke([system_message, *history])

        updates: dict[str, Any] = {
            "messages": [response],
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
            "depth_mode": depth_mode,
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
            "metrics": {"plan_length": len(plan), "steps": int(state.get("iteration_count", 0)) + 1},
        }

        tool_names = _extract_tool_names(response)
        if tool_names:
            updates["tool_calls_made"] = [*state.get("tool_calls_made", []), *tool_names]
        else:
            content = str(response.content or "").strip()
            if content:
                updates["final_answer"] = content
                updates["final_answer_reviewed"] = False

        return updates

    return agent_node, tools


def _build_review_node(model_name: str, temperature: float):
    def review_node(state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        draft_answer = str(state.get("final_answer") or "").strip()
        if not draft_answer:
            draft_answer = _best_available_answer_from_messages(messages)

        if not draft_answer:
            return {"final_answer_reviewed": True}

        query = _extract_user_query(messages)
        intent = str(state.get("intent", INTENT_DISCOVERY))
        answer_type = _resolve_answer_type(state)
        depth_mode = _resolve_depth_mode(state)
        evidence = _collect_evidence(messages)
        issues = _answer_quality_issues(
            draft_answer,
            answer_type,
            intent,
            depth_mode,
            list(state.get("tool_calls_made", [])),
        )

        if answer_type in {ANSWER_AMBIGUOUS, ANSWER_CALCULATION, ANSWER_MULTI}:
            return {
                "messages": [AIMessage(content=draft_answer)],
                "final_answer": draft_answer,
                "final_answer_reviewed": True,
                "depth_mode": depth_mode,
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
- Return nothing except the correctly formatted answer.
""".format(
                    response_template=_build_response_template(intent, depth_mode, answer_type),
                    issues="\n".join(f"- {issue}" for issue in issues) if issues else "- None",
                    query=query,
                    intent=intent,
                    answer_type=answer_type,
                    depth_mode=depth_mode,
                    draft_answer=draft_answer,
                    evidence=evidence,
                )
            ),
            HumanMessage(content="Rewrite the draft and return only the final answer in the required format."),
        ]

        revised_answer = draft_answer
        for _ in range(2):
            response = model.invoke(prompt)
            candidate_answer = str(response.content or "").strip()
            if candidate_answer:
                revised_answer = candidate_answer
            issues = _answer_quality_issues(
                revised_answer,
                answer_type,
                intent,
                depth_mode,
                list(state.get("tool_calls_made", [])),
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

Depth mode:
{depth_mode}

Draft answer:
{draft_answer}

Evidence:
{evidence}
""".format(
                    response_template=_build_response_template(intent, depth_mode, answer_type),
                    issues="\n".join(f"- {issue}" for issue in issues),
                    query=query,
                    intent=intent,
                    answer_type=answer_type,
                    depth_mode=depth_mode,
                    draft_answer=revised_answer,
                    evidence=evidence,
                )
            )

        return {
            "messages": [AIMessage(content=revised_answer)],
            "final_answer": revised_answer,
            "final_answer_reviewed": True,
            "depth_mode": depth_mode,
            "validation_errors": issues,
            "answer_format_ok": not issues,
        }

    return review_node


def _build_validate_node():
    def validate_node(state: AgentState) -> dict[str, Any]:
        answer = str(state.get("final_answer") or "").strip()
        messages = list(state.get("messages", []))
        intent = str(state.get("intent", INTENT_DISCOVERY))
        answer_type = _resolve_answer_type(state)
        depth_mode = _resolve_depth_mode(state)
        issues = _answer_quality_issues(
            answer,
            answer_type,
            intent,
            depth_mode,
            list(state.get("tool_calls_made", [])),
        )

        if not answer:
            issues.append("empty answer")

        retry_count = int(state.get("retry_count", 0))
        if issues and retry_count < MAX_RETRIES:
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
                },
            )

        return {
            "needs_retry": False,
            "validation_errors": issues,
            "answer_format_ok": not issues,
        }

    return validate_node


def _best_available_answer_from_messages(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = str(message.content or "").strip()
            if content:
                return content

    observations: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            content = str(message.content or "").strip()
            if content:
                observations.append(content)

    if observations:
        return observations[-1]

    return ""


def _should_continue(state: AgentState) -> str:
    max_iterations = int(state.get("max_iterations", 5))
    if int(state.get("iteration_count", 0)) >= max_iterations:
        return END

    if (
        str(state.get("confidence") or "").lower() == "high"
        and bool(state.get("final_answer_reviewed", False))
        and not bool(state.get("needs_retry", False))
    ):
        return END

    messages = list(state.get("messages", []))
    if not messages:
        return END

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"

    final_answer = str(state.get("final_answer") or "").strip()
    if isinstance(last_message, AIMessage) and final_answer:
        if not bool(state.get("final_answer_reviewed", False)):
            return "review"
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
    review_node = _build_review_node(resolved_model_name, resolved_temperature)
    validate_node = _build_validate_node()
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("review", review_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", "review": "review", END: END})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("review", "validate")
    workflow.add_conditional_edges("validate", _should_retry_after_validation, {"agent": "agent", END: END})
    return workflow.compile()
