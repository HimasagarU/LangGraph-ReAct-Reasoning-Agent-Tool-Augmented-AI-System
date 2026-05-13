from __future__ import annotations

import ast
import json
import math
import os
import operator
from urllib.parse import quote
from urllib.parse import urlparse
from typing import Any

import logging

import requests
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import BaseTool, tool
from tavily import TavilyClient

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_WIKIPEDIA_HEADERS = {
    "User-Agent": "LangGraphReActAgent/1.0 (https://github.com/himas/langgraph-react-agent)",
    "Accept": "application/json",
}

_PAGE_FETCH_HEADERS = {
    "User-Agent": "LangGraphReActAgent/1.0 (https://github.com/himas/langgraph-react-agent)",
}

def _handle_tool_error(tool_name: str, exc: Exception) -> str:
    """Centralized tool error handling."""
    error_msg = str(exc)
    logging.warning(f"[{tool_name}] failed: {error_msg}")
    if isinstance(exc, requests.exceptions.Timeout):
        return f"{tool_name} failed: Request timed out."
    return f"{tool_name} failed: {error_msg}"


def _stringify_tool_output(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _clean_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _extract_page_title(html: str) -> str:
    """Extract title from HTML using simple regex."""
    import re
    match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if match:
        title = match.group(1).strip()
        return title[:120]
    return "Page content"


def _extract_main_text(html: str) -> str:
    """Extract main readable text from HTML by removing script/style tags and markup."""
    import re
    if not html:
        return ""
    
    # Remove script and style tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove HTML tags but keep content
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    
    # Collapse whitespace
    text = " ".join(text.split())
    
    return text


def score_source_quality(url: str, snippet: str = "", title: str = "") -> int:
    """Score a source by rough trustworthiness for ranking and confidence heuristics."""
    domain = urlparse(url).netloc.lower()
    combined = f"{title} {snippet}".lower().strip()
    score = 0

    if domain.endswith(".gov") or domain.endswith(".edu") or domain.endswith(".ac"):
        score += 5
    if any(token in domain for token in ["arxiv.org", "who.int", "un.org", "europa.eu"]):
        score += 5
    if any(token in combined for token in ["official", "report", "press release", "whitepaper"]):
        score += 2
    if url.lower().endswith(".pdf"):
        score += 2
    if "wikipedia.org" in domain:
        score += 3
    if any(token in domain for token in ["medium.com", "blog", "substack.com", "reddit.com", "quora.com", "forum"]):
        score -= 2

    return score


def _format_tavily_results(payload: Any) -> str:
    results = payload.get("results") if isinstance(payload, dict) and isinstance(payload.get("results"), list) else []
    if not results:
        return json.dumps({"results": []}, ensure_ascii=False)

    structured_results: list[dict[str, Any]] = []
    for result in results[:3]:
        title = _clean_text(str(result.get("title") or "").strip(), 120)
        url = str(result.get("url") or "").strip()
        snippet = _clean_text(str(result.get("content") or result.get("snippet") or "").strip(), 220)
        if not url:
            continue
        structured_results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "score": score_source_quality(url, snippet=snippet, title=title),
            }
        )

    structured_results.sort(key=lambda item: item["score"], reverse=True)
    return json.dumps({"results": structured_results}, ensure_ascii=False)


def _invoke_tool(tool_object: Any, query: str) -> Any:
    if hasattr(tool_object, "invoke"):
        try:
            return tool_object.invoke(query)
        except TypeError:
            return tool_object.invoke({"query": query})
    if hasattr(tool_object, "run"):
        return tool_object.run(query)
    raise TypeError("Tool object does not expose invoke() or run().")


def _safe_eval(expression: str) -> float:
    parsed = ast.parse(expression, mode="eval")

    def _evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            left_value = _evaluate(node.left)
            right_value = _evaluate(node.right)
            return float(_ALLOWED_OPERATORS[type(node.op)](left_value, right_value))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            operand_value = _evaluate(node.operand)
            return float(_ALLOWED_OPERATORS[type(node.op)](operand_value))
        raise ValueError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")

    result = _evaluate(parsed)
    if not math.isfinite(result):
        raise ValueError("Expression did not evaluate to a finite number")
    return result


def _fallback_wikipedia_summary(query: str, return_title: bool = False) -> str | tuple[str, str]:
    search_response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        headers=_WIKIPEDIA_HEADERS,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
        },
        timeout=5,
    )
    search_response.raise_for_status()
    search_payload = search_response.json()
    search_hits = search_payload.get("query", {}).get("search", [])
    if not search_hits:
        if return_title:
            return "", "Wikipedia lookup failed: no results found."
        return "Wikipedia lookup failed: no results found."

    page_title = search_hits[0].get("title", "")
    if not page_title:
        if return_title:
            return "", "Wikipedia lookup failed: no title found for the search result."
        return "Wikipedia lookup failed: no title found for the search result."

    summary_response = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(page_title)}",
        headers=_WIKIPEDIA_HEADERS,
        timeout=5,
    )
    summary_response.raise_for_status()
    summary_payload = summary_response.json()

    extract = str(summary_payload.get("extract") or "").strip()
    if extract:
        if return_title:
            return page_title, extract
        return extract

    description = str(summary_payload.get("description") or "").strip()
    if description:
        if return_title:
            return page_title, description
        return description

    summary = f"Wikipedia summary unavailable for {page_title}."
    if return_title:
        return page_title, summary
    return summary


def _build_wikipedia_fallback_result(query: str) -> dict[str, Any] | None:
    try:
        title, summary = _fallback_wikipedia_summary(query, return_title=True)
    except Exception:
        return None

    if not title:
        return None

    url_title = quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/wiki/{url_title}"
    snippet = _clean_text(summary, 220)
    if not snippet or "failed" in snippet.lower():
        return None
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "score": score_source_quality(url, snippet=snippet, title=title),
    }


@tool("tavily_search")
def tavily_search(query: str) -> str:
    """Search the web for current information and return the top results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps({"results": [], "error": "Tavily search is unavailable because TAVILY_API_KEY is not set."})

    try:
        search_client = TavilyClient(api_key=api_key)
        result = search_client.search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=3,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        )
    except Exception as exc:
        return json.dumps({"results": [], "error": _handle_tool_error("Tavily search", exc)})

    if isinstance(result, dict) and not result.get("results"):
        fallback = _build_wikipedia_fallback_result(query)
        if fallback:
            return json.dumps({"results": [fallback], "fallback": "wikipedia_lookup"}, ensure_ascii=False)
        return json.dumps({"results": [], "error": "Tavily search returned no results."}, ensure_ascii=False)

    return _format_tavily_results(result)


@tool("wikipedia_lookup")
def wikipedia_lookup(query: str) -> str:
    """Look up concise encyclopedic information from Wikipedia."""
    wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    try:
        result = _invoke_tool(wikipedia_tool, query)
        result_text = _stringify_tool_output(result).strip()
        if result_text:
            return _clean_text(result_text, 420)
    except Exception:
        pass

    try:
        return _clean_text(_fallback_wikipedia_summary(query), 420)
    except Exception as exc:
        return _handle_tool_error("Wikipedia lookup", exc)


@tool("calculator")
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    allowed_characters = set("0123456789+-*/()., ")
    def _format_calc_error(msg: str) -> str:
        return json.dumps({"results": [{"title": "Calculation Error", "url": "calculator", "snippet": msg}]})

    if not expression:
        return _format_calc_error("Error: expression is empty")
    if any(character not in allowed_characters for character in expression):
        return _format_calc_error("Error: invalid characters in expression")

    try:
        result = round(_safe_eval(expression), 6)
        if float(result).is_integer():
            final_res = str(int(result))
        else:
            final_res = str(result)
        return json.dumps({"results": [{"title": "Calculation Result", "url": "calculator", "snippet": final_res}]})
    except Exception as exc:
        return _format_calc_error(f"Error: {exc}")


@tool("page_fetch")
def page_fetch(url: str) -> str:
    """Fetch and parse the main content from a web page given its URL."""
    if not url:
        return json.dumps({"results": [], "error": "URL is required."})

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        response = requests.get(
            url,
            headers=_PAGE_FETCH_HEADERS,
            timeout=8,
            allow_redirects=True,
            stream=False,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return json.dumps({"results": [], "error": "Request timed out."})
    except requests.exceptions.RequestException as exc:
        return json.dumps({"results": [], "error": f"Failed to fetch URL: {str(exc)}"})

    try:
        html = response.text
        if not html or len(html) < 100:
            return json.dumps({"results": [], "error": "Page returned no content or was too short."})

        title = _extract_page_title(html)
        content = _extract_main_text(html)

        if not content or len(content) < 50:
            return json.dumps({"results": [], "error": "Could not extract meaningful content from page."})

        snippet = _clean_text(content, 420)
        quality_score = score_source_quality(url, snippet=snippet, title=title)

        return json.dumps(
            {
                "results": [
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "score": quality_score,
                    }
                ]
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"results": [], "error": _handle_tool_error("Page fetch", exc)})


def build_tools() -> list[BaseTool]:
    return [tavily_search, wikipedia_lookup, calculator, page_fetch]
