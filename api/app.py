from __future__ import annotations

import asyncio
import json
import os
import queue
import re
from urllib.parse import urlparse
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.graph import build_graph
from agent.tools import score_source_quality
import httpx

load_dotenv()

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
DEFAULT_MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
DONE_SENTINEL = object()

app = FastAPI(
    title="LangGraph ReAct Agent",
    version="0.1.0",
    description="A LangGraph ReAct agent with intent-aware tool routing and SSE streaming.",
)

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="User question or prompt.")
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1, le=10)
    model_name: str | None = Field(default=None, description="Optional Groq model override.")
    reasoning_budget: str | None = Field(default=None, description="Reasoning depth: shallow, medium, deep, or auto (default).")
    # Deprecation shim: accept old depth_mode values
    depth_mode: str | None = Field(default=None, description="Deprecated. Use reasoning_budget instead.")

    def effective_budget(self) -> str | None:
        """Resolve reasoning_budget, with backward compat for depth_mode."""
        if self.reasoning_budget:
            return self.reasoning_budget
        if self.depth_mode:
            _SHIM = {"concise": "shallow", "standard": "medium", "learning_ml": "deep"}
            return _SHIM.get(self.depth_mode, self.depth_mode)
        return None


class TraceStep(BaseModel):
    thought: str
    action: str
    observation: str | None = None


class SourceItem(BaseModel):
    title: str | None = None
    url: str
    snippet: str | None = None


class QueryResponse(BaseModel):
    answer: str
    intent: str
    tools_used: list[str]
    iterations: int
    latency_ms: float
    trace: list[TraceStep]
    confidence: str | None = None
    answer_type: str | None = None
    plan: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    tools: list[str]
    model: str
    dependencies: dict[str, bool]


class _TokenQueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, token_queue: "queue.Queue[str | object]") -> None:
        self._token_queue = token_queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # pragma: no cover - callback interface
        if token:
            self._token_queue.put(token)


def _resolve_model_name(explicit_model_name: str | None = None) -> str:
    return explicit_model_name or DEFAULT_MODEL_NAME

def _resolve_temperature() -> float:
    try:
        return float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    except ValueError:
        return 0.2


@lru_cache(maxsize=8)
def _compiled_graph(model_name: str, temperature: float):
    return build_graph(model_name=model_name, temperature=temperature)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


_URL_PATTERN = re.compile(r"https?://[^\s)]+")
def _clean_snippet(snippet: str) -> str | None:
    cleaned = " ".join(snippet.split()).strip()
    if not cleaned:
        return None
    if len(cleaned) > 240:
        return cleaned[:237].rstrip() + "..."
    return cleaned


def _extract_sources(messages: list[Any]) -> list[dict[str, str | None]]:
    collected: list[dict[str, str | None]] = []
    seen: set[str] = set()
    order = 0

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content or "")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            for item in payload.get("results", []):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                key = url.lower()
                if key in seen or "localhost" in key or "127.0.0.1" in key:
                    continue
                title = str(item.get("title") or "").strip() or None
                snippet = _clean_snippet(str(item.get("snippet") or ""))
                score = int(item.get("score") or score_source_quality(url, snippet=snippet or "", title=title or ""))
                collected.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "score": score,
                        "order": order,
                    }
                )
                seen.add(key)
                order += 1
            continue

        for line in content.splitlines():
            match = _URL_PATTERN.search(line)
            if not match:
                continue
            url = match.group(0).rstrip(".,;")
            key = url.lower()
            if key in seen or "localhost" in key or "127.0.0.1" in key:
                continue
            title = line[: match.start()].strip(" -")
            if title.lower().startswith("top results"):
                title = ""
            snippet = _clean_snippet(line[match.end() :].strip(" -"))
            score = score_source_quality(url, snippet=snippet or "", title=title)
            collected.append(
                {
                    "title": title or None,
                    "url": url,
                    "snippet": snippet,
                    "score": score,
                    "order": order,
                }
            )
            seen.add(key)
            order += 1

    collected.sort(key=lambda item: (-int(item.get("score", 0)), int(item.get("order", 0))))
    return [
        {
            "title": item.get("title"),
            "url": str(item.get("url")),
            "snippet": item.get("snippet"),
        }
        for item in collected[:4]
    ]


def _extract_trace(messages: list[Any], final_answer_reviewed: bool = False) -> list[dict[str, str | None]]:
    trace: list[dict[str, str | None]] = []

    for message in messages:
        if isinstance(message, AIMessage):
            thought = str(message.content or "").strip()
            tool_calls = getattr(message, "tool_calls", []) or []
            if tool_calls:
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = str(tool_call.get("name", "tool"))
                    else:
                        tool_name = str(getattr(tool_call, "name", "tool"))
                    trace.append(
                        {
                            "thought": thought or "Tool use requested.",
                            "action": tool_name or "tool",
                            "observation": None,
                        }
                    )
            else:
                trace.append(
                    {
                        "thought": thought or "I have enough information to answer.",
                        "action": "FINISH",
                        "observation": None,
                    }
                )
        elif isinstance(message, ToolMessage):
            observation = str(message.content or "").strip() or None
            for step in reversed(trace):
                if step["action"] != "FINISH" and step["observation"] is None:
                    step["observation"] = observation
                    break

    deduped_trace: list[dict[str, str | None]] = []
    for step in trace:
        if (
            deduped_trace
            and step["action"] == "FINISH"
            and deduped_trace[-1]["action"] == "FINISH"
            and step.get("observation") == deduped_trace[-1].get("observation")
        ):
            deduped_trace[-1] = step
            continue
        deduped_trace.append(step)

    if final_answer_reviewed and len(deduped_trace) >= 2:
        if deduped_trace[-1]["action"] == "FINISH" and deduped_trace[-2]["action"] == "FINISH":
            deduped_trace.pop(-2)

    for step in deduped_trace:
        if step["action"] == "FINISH":
            step["thought"] = "I have enough information to answer."

    return deduped_trace


def _best_available_answer(state: dict[str, Any]) -> str:
    final_answer = str(state.get("final_answer") or "").strip()
    messages = list(state.get("messages", []))

    retry_cutoff = -1
    for index, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            content = str(message.content or "")
            if "Validation failed. Retry" in content:
                retry_cutoff = index

    if final_answer and (bool(state.get("final_answer_reviewed", False)) or retry_cutoff < 0):
        return final_answer

    for message in reversed(messages[retry_cutoff + 1 :]):
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

    return "I could not complete the task with the available evidence."


def _build_initial_state(query: str, reasoning_budget: str | None, max_iterations: int) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=query)],
        "reasoning_budget": reasoning_budget or "",
        "tool_calls_made": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "confidence": "low",
        "rewritten_query": "",
        "rewrite_variants": [],
        "retry_count": 0,
        "needs_retry": False,
        "final_answer": "",
        "final_answer_reviewed": False,
        "validation_errors": [],
        "answer_format_ok": False,
        "evidence_pack": {},
        "critic_issues": [],
    }


def _run_agent_sync(
    query: str,
    max_iterations: int,
    model_name: str,
    reasoning_budget: str | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    graph = _compiled_graph(model_name, _resolve_temperature())
    state = graph.invoke(
        _build_initial_state(
            query=query,
            reasoning_budget=reasoning_budget,
            max_iterations=max_iterations,
        ),
        config={"callbacks": callbacks or []},
    )

    if not isinstance(state, dict):
        raise RuntimeError("Agent graph returned an unexpected state payload.")

    answer = _best_available_answer(state)
    tools_used = _dedupe_preserve_order([str(name) for name in state.get("tool_calls_made", []) if name])
    trace = _extract_trace(list(state.get("messages", [])), bool(state.get("final_answer_reviewed", False)))
    sources = _extract_sources(list(state.get("messages", [])))
    confidence = str(state.get("confidence") or "").strip() or None
    plan = [str(step) for step in state.get("plan", []) if str(step).strip()]
    metrics = {
        "steps": int(state.get("iteration_count", 0)),
        "tools_used": tools_used,
        "retry_count": int(state.get("retry_count", 0)),
        "confidence": confidence,
        "reasoning_budget": str(state.get("reasoning_budget", "medium")),
    }

    return {
        "answer": answer,
        "intent": str(state.get("intent", "discovery")),
        "tools_used": tools_used,
        "iterations": int(state.get("iteration_count", 0)),
        "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
        "trace": trace,
        "confidence": confidence,
        "answer_type": str(state.get("answer_type", "explanation")).strip() or None,
        "plan": plan,
        "metrics": metrics,
        "validation_errors": [str(item) for item in state.get("validation_errors", []) if str(item).strip()],
        "sources": sources,
    }


@app.head("/ping")
@app.get("/ping")
async def ping() -> dict[str, str]:
    """Ultra-lightweight ping endpoint for cron jobs. No dependencies checked."""
    return {"status": "alive"}


async def _keep_alive_loop() -> None:
    """Background task to ping the server itself to prevent Render spin-down."""
    external_url = os.getenv("RENDER_EXTERNAL_URL")
    if not external_url:
        # Fallback to RENDER_URL if EXTERNAL is not set
        external_url = os.getenv("RENDER_URL")
        
    if not external_url:
        print("KEEP_ALIVE: No RENDER_EXTERNAL_URL or RENDER_URL set. Self-ping disabled.")
        return

    # Normalize URL: ensure it starts with http and doesn't have trailing slash for the join
    external_url = external_url.rstrip("/")
    if not external_url.startswith("http"):
        external_url = f"https://{external_url}"
        
    ping_url = f"{external_url}/ping"
    print(f"KEEP_ALIVE: Starting background self-ping for {ping_url}")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            # Render spins down after 15 mins of inactivity. 
            # Ping every 10 minutes to stay safely within the window.
            await asyncio.sleep(600)
            try:
                resp = await client.get(ping_url)
                print(f"KEEP_ALIVE: Self-ping status={resp.status_code}")
            except Exception as e:
                print(f"KEEP_ALIVE: Self-ping failed: {e}")


@app.on_event("startup")
async def on_startup():
    # Start the keep-alive loop in the background
    asyncio.create_task(_keep_alive_loop())


@app.head("/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        tools=["tavily_search", "wikipedia_lookup", "calculator", "page_fetch"],
        model=_resolve_model_name(),
        dependencies={
            "groq_api_key": bool(os.getenv("GROQ_API_KEY")),
            "tavily_api_key": bool(os.getenv("TAVILY_API_KEY")),
        },
    )


@app.head("/styles.css", include_in_schema=False)
@app.get("/styles.css", include_in_schema=False)
async def frontend_styles() -> FileResponse:
    css_file = FRONTEND_DIR / "styles.css"
    if not css_file.exists():
        raise HTTPException(status_code=404, detail="CSS asset not available.")
    return FileResponse(css_file, media_type="text/css")


@app.head("/app.js", include_in_schema=False)
@app.get("/app.js", include_in_schema=False)
async def frontend_script() -> FileResponse:
    js_file = FRONTEND_DIR / "app.js"
    if not js_file.exists():
        raise HTTPException(status_code=404, detail="JS asset not available.")
    return FileResponse(js_file, media_type="application/javascript")


@app.head("/", include_in_schema=False)
@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend assets are not available.")
    return FileResponse(index_file)


@app.post("/agent/query", response_model=QueryResponse)
async def agent_query(request: QueryRequest) -> QueryResponse:
    try:
        result = await run_in_threadpool(
            _run_agent_sync,
            request.query,
            request.max_iterations,
            _resolve_model_name(request.model_name),
            request.effective_budget(),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

    return QueryResponse(**result)


async def _stream_agent_events(request: QueryRequest) -> AsyncIterator[str]:
    token_queue: "queue.Queue[str | object]" = queue.Queue()
    callback_handler = _TokenQueueCallbackHandler(token_queue)
    result_box: dict[str, Any] = {}
    done_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _worker() -> None:
        try:
            result_box["result"] = _run_agent_sync(
                request.query,
                request.max_iterations,
                _resolve_model_name(request.model_name),
                request.effective_budget(),
                callbacks=[callback_handler],
            )
        except Exception as exc:  # pragma: no cover - background worker path
            result_box["error"] = exc
        finally:
            token_queue.put(DONE_SENTINEL)
            loop.call_soon_threadsafe(done_event.set)

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()

    while True:
        try:
            token = await asyncio.wait_for(asyncio.to_thread(token_queue.get), timeout=90.0)
        except asyncio.TimeoutError:
            if worker_thread.is_alive():
                worker_thread.join(timeout=3.0)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Request timeout'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        if token is DONE_SENTINEL:
            break
        yield f"data: {json.dumps({'type': 'token', 'text': token}, ensure_ascii=False)}\n\n"

    try:
        await asyncio.wait_for(done_event.wait(), timeout=90.0)
    except asyncio.TimeoutError:
        if worker_thread.is_alive():
            worker_thread.join(timeout=3.0)
        yield f"data: {json.dumps({'type': 'error', 'message': 'Request timeout'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return
    except asyncio.CancelledError:
        if worker_thread.is_alive():
            worker_thread.join(timeout=3.0)
        raise
    finally:
        if worker_thread.is_alive():
            worker_thread.join(timeout=3.0)

    if "error" in result_box:
        yield f"data: {json.dumps({'type': 'error', 'message': str(result_box['error'])}, ensure_ascii=False)}\n\n"
    else:
        result = result_box.get("result")
        yield f"data: {json.dumps({'type': 'final', 'result': result}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/agent/stream")
async def agent_stream(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(_stream_agent_events(request), media_type="text/event-stream")
