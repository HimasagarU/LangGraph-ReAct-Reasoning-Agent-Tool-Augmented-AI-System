from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import _resolve_model_name, _run_agent_sync  # noqa: E402


@dataclass
class CaseResult:
    case_id: str
    query: str
    passed: bool
    checks: dict[str, bool]
    details: dict[str, Any]
    skipped: bool = False
    response: dict[str, Any] | None = None
    error: str | None = None


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = line.strip()
        if not payload:
            continue
        cases.append(json.loads(payload))
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def _has_basic_format(answer_type: str, answer: str) -> bool:
    lowered = answer.lower()
    if answer_type == "ambiguous":
        return answer.strip().endswith("?") and answer.count("?") == 1
    if answer_type == "calculation":
        return bool(re.search(r"\d", answer))
    if answer_type == "comparison":
        return "|" in answer and "verdict" in lowered
    if answer_type == "fact":
        return "**answer:**" in lowered or bool(answer.strip())
    if answer_type == "list":
        return bool(re.search(r"(?m)^(?:[-*]\s+|\d+\.)", answer))
    if answer_type == "multi":
        return "### explanation" in lowered and "### calculation" in lowered
    return bool(answer.strip())


def _evaluate_case(case: dict[str, Any], response: dict[str, Any]) -> CaseResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    expected_intent = str(case.get("expected_intent") or "").strip()
    expected_answer_type = str(case.get("expected_answer_type") or "").strip()
    required_tools = [str(item) for item in case.get("required_tools", [])]
    required_any_tools = [str(item) for item in case.get("required_any_tools", [])]
    must_contain = [str(item) for item in case.get("must_contain", [])]
    must_not_contain = [str(item) for item in case.get("must_not_contain", [])]

    answer = str(response.get("answer") or "")
    tools_used = [str(tool) for tool in response.get("tools_used", [])]
    intent = str(response.get("intent") or "")
    answer_type = str(response.get("answer_type") or "")

    checks["non_empty_answer"] = bool(answer.strip())
    if expected_intent:
        checks["intent_match"] = intent == expected_intent
    if expected_answer_type:
        checks["answer_type_match"] = answer_type == expected_answer_type
        checks["answer_format"] = _has_basic_format(expected_answer_type, answer)
    if required_tools:
        checks["required_tools_used"] = all(tool in tools_used for tool in required_tools)
    if required_any_tools:
        checks["required_any_tool_used"] = any(tool in tools_used for tool in required_any_tools)
    if must_contain:
        checks["contains_expected_text"] = all(token.lower() in answer.lower() for token in must_contain)
    if must_not_contain:
        checks["avoids_forbidden_text"] = all(token.lower() not in answer.lower() for token in must_not_contain)

    passed = all(checks.values()) if checks else False
    details["intent"] = intent
    details["answer_type"] = answer_type
    details["tools_used"] = tools_used
    details["latency_ms"] = response.get("latency_ms")

    return CaseResult(
        case_id=str(case.get("id") or case.get("query") or "case"),
        query=str(case.get("query") or ""),
        passed=passed,
        checks=checks,
        details=details,
        response=response,
    )


def _run_benchmark(
    cases: list[dict[str, Any]],
    model_name: str | None,
    max_iterations: int,
    reasoning_budget: str | None,
) -> list[CaseResult]:
    resolved_model_name = _resolve_model_name(model_name)
    results: list[CaseResult] = []
    for case in cases:
        query = str(case.get("query") or "").strip()
        requires_llm = bool(case.get("requires_llm", True))
        if not query:
            results.append(
                CaseResult(
                    case_id=str(case.get("id") or "unknown"),
                    query="",
                    passed=False,
                    checks={"valid_query": False},
                    details={},
                    skipped=False,
                    error="Empty query in evaluation case.",
                )
            )
            continue

        try:
            response = _run_agent_sync(
                query=query,
                max_iterations=max_iterations,
                model_name=resolved_model_name,
                reasoning_budget=reasoning_budget,
            )
            results.append(_evaluate_case(case, response))
        except Exception as exc:
            error_text = str(exc)
            skipped_llm_errors = (
                "GROQ_API_KEY is required",
                "Connection error",
                "timed out",
                "401",
            )
            if requires_llm and any(token in error_text for token in skipped_llm_errors):
                results.append(
                    CaseResult(
                        case_id=str(case.get("id") or query),
                        query=query,
                        passed=False,
                        checks={"skipped_missing_llm_key": True},
                        details={},
                        skipped=True,
                        error=error_text,
                    )
                )
                continue
            results.append(
                CaseResult(
                    case_id=str(case.get("id") or query),
                    query=query,
                    passed=False,
                    checks={"execution_success": False},
                    details={},
                    skipped=False,
                    error=str(exc),
                )
            )
    return results


def _summarize(results: list[CaseResult]) -> dict[str, Any]:
    skipped_cases = sum(1 for result in results if result.skipped)
    evaluated_results = [result for result in results if not result.skipped]
    total_cases = len(evaluated_results)
    passed_cases = sum(1 for result in evaluated_results if result.passed)
    check_pass_tally: dict[str, int] = defaultdict(int)
    check_total_tally: dict[str, int] = defaultdict(int)

    for result in evaluated_results:
        for name, passed in result.checks.items():
            check_total_tally[name] += 1
            if passed:
                check_pass_tally[name] += 1

    check_breakdown = {
        name: {
            "passed": check_pass_tally[name],
            "total": check_total_tally[name],
            "pass_rate": round(check_pass_tally[name] / check_total_tally[name], 3),
        }
        for name in sorted(check_total_tally)
    }

    return {
        "total_cases": total_cases,
        "skipped_cases": skipped_cases,
        "passed_cases": passed_cases,
        "pass_rate": round(passed_cases / total_cases, 3) if total_cases else 0.0,
        "check_breakdown": check_breakdown,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark evals for the LangGraph ReAct agent.")
    parser.add_argument("--cases", type=Path, default=Path("evals/benchmark_cases.jsonl"), help="Path to JSONL evaluation cases.")
    parser.add_argument("--output", type=Path, default=Path("evals/latest_eval_report.json"), help="Where to write evaluation results.")
    parser.add_argument("--model-name", type=str, default=None, help="Optional model override.")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max LangGraph loop iterations per case.")
    parser.add_argument("--reasoning-budget", type=str, default=None, help="Optional fixed reasoning budget.")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    results = _run_benchmark(
        cases=cases,
        model_name=args.model_name,
        max_iterations=args.max_iterations,
        reasoning_budget=args.reasoning_budget,
    )
    summary = _summarize(results)
    payload = {
        "summary": summary,
        "results": [
            {
                "case_id": result.case_id,
                "query": result.query,
                "passed": result.passed,
                "skipped": result.skipped,
                "checks": result.checks,
                "details": result.details,
                "error": result.error,
            }
            for result in results
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Cases passed: {summary['passed_cases']}/{summary['total_cases']} ({summary['pass_rate']:.1%})")
    if summary["skipped_cases"]:
        print(f"Cases skipped: {summary['skipped_cases']}")
    print(f"Report: {args.output}")
    for check_name, check_stats in summary["check_breakdown"].items():
        print(f"- {check_name}: {check_stats['passed']}/{check_stats['total']} ({check_stats['pass_rate']:.1%})")

    return 1 if summary["passed_cases"] < summary["total_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
