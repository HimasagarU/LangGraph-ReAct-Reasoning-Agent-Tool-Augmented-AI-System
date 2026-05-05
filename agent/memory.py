from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

# ── Memory toggle ─────────────────────────────────────────────────────────────
# Set MEMORY_ENABLED=false in env to disable memory writes entirely
# (useful for ephemeral filesystems like Render free tier).
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() in {"true", "1", "yes"}

_MEMORY_PATH = Path(__file__).resolve().parents[1] / ".agent_memory.jsonl"
_MEMORY_LOCK = threading.Lock()
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MAX_RECORDS = 500


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.lower()) if len(token) > 2}


def _load_records() -> list[dict[str, Any]]:
    if not _MEMORY_PATH.exists():
        return []

    records: list[dict[str, Any]] = []
    try:
        with _MEMORY_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except OSError:
        return []

    return records


def recall_memory(query: str, budget: str = "shallow", limit: int = 3) -> list[dict[str, Any]]:
    """Return the most relevant prior interactions for a query.

    Budget controls recall depth:
    - shallow: disabled (returns empty)
    - medium: recall 1 record
    - deep: recall up to `limit` records (default 3)
    """
    if not MEMORY_ENABLED:
        return []

    if budget == "shallow":
        return []

    effective_limit = 1 if budget == "medium" else limit

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored_records: list[tuple[int, dict[str, Any]]] = []
    for record in _load_records():
        prior_query = str(record.get("query") or "")
        prior_answer = str(record.get("answer") or "")
        record_tokens = _tokenize(f"{prior_query} {prior_answer}")
        overlap = len(query_tokens & record_tokens)
        if overlap == 0:
            continue
        scored_records.append((overlap, record))

    scored_records.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored_records[:effective_limit]]


def format_memory_context(query: str, budget: str = "shallow", limit: int = 3) -> str:
    """Format recent relevant memory as a prompt-friendly context block.

    Budget controls recall depth (see recall_memory).
    """
    if not MEMORY_ENABLED:
        return ""

    if budget == "shallow":
        return ""

    records = recall_memory(query, budget=budget, limit=limit)
    if not records:
        return ""

    lines = ["Previous similar interactions:"]
    for record in records:
        prior_query = str(record.get("query") or "").strip()
        prior_answer = str(record.get("answer") or "").strip()
        if not prior_query and not prior_answer:
            continue
        parts = []
        if prior_query:
            parts.append(f"Q: {prior_query}")
        if prior_answer:
            parts.append(f"A: {prior_answer[:260]}")
        lines.append(" - " + " | ".join(parts))
    return "\n".join(lines)


def remember_interaction(
    query: str,
    answer: str,
    metadata: dict[str, Any] | None = None,
    budget: str = "shallow",
) -> None:
    """Persist a lightweight interaction record for later recall.

    Budget controls storage:
    - shallow: disabled (no write)
    - medium: store basic record
    - deep: store record with failure_mode metadata
    """
    if not MEMORY_ENABLED:
        return

    if budget == "shallow":
        return

    payload: dict[str, Any] = {
        "query": query.strip(),
        "answer": answer.strip(),
        "metadata": metadata or {},
    }

    try:
        _MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _MEMORY_LOCK:
            records = _load_records()
            records.append(payload)
            if len(records) > MAX_RECORDS:
                records = records[-MAX_RECORDS:]
                
            fd, temp_path = tempfile.mkstemp(dir=str(_MEMORY_PATH.parent), prefix=".agent_memory_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                    for record in records:
                        temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                os.replace(temp_path, str(_MEMORY_PATH))
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    except OSError:
        return
