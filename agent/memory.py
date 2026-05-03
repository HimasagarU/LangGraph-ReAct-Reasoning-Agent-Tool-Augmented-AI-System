from __future__ import annotations

import json
import re
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

ENABLE_MEMORY = False

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


def recall_memory(query: str, limit: int = 2) -> list[dict[str, Any]]:
    """Return the most relevant prior interactions for a query."""
    if not ENABLE_MEMORY:
        return []
        
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
    return [record for _, record in scored_records[:limit]]


def format_memory_context(query: str, limit: int = 2) -> str:
    """Format recent relevant memory as a prompt-friendly context block."""
    if not ENABLE_MEMORY:
        return ""
        
    records = recall_memory(query, limit=limit)
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


def remember_interaction(query: str, answer: str, metadata: dict[str, Any] | None = None) -> None:
    """Persist a lightweight interaction record for later recall."""
    if not ENABLE_MEMORY:
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
