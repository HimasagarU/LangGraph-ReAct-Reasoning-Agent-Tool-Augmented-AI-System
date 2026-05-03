from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from api.app import _best_available_answer, _extract_sources, _extract_trace, app

FAKE_RESULT = {
    "answer": "RAG stands for Retrieval-Augmented Generation.",
    "intent": "explanatory",
    "answer_type": "explanation",
    "tools_used": ["wikipedia_lookup"],
    "iterations": 2,
    "latency_ms": 123.45,
    "trace": [
        {
            "thought": "The user wants a definition.",
            "action": "wikipedia_lookup",
            "observation": "RAG is a technique that combines retrieval and generation.",
        },
        {
            "thought": "I have enough information to answer.",
            "action": "FINISH",
            "observation": None,
        },
    ],
    "confidence": "medium",
    "plan": ["search for the fact", "extract the direct value", "return value source confidence"],
    "metrics": {"steps": 2, "tools_used": ["wikipedia_lookup"], "retry_count": 0, "confidence": "medium"},
    "validation_errors": [],
    "sources": [
        {
            "title": "Retrieval-augmented generation",
            "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            "snippet": "Retrieval-augmented generation combines retrieval with text generation.",
        }
    ],
}


class FrontendSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_root_serves_frontend(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("LangGraph ReAct Agent", response.text)
        self.assertIn("query-form", response.text)
        self.assertIn("POST /agent/query", response.text)

    def test_health_endpoint_returns_status(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("tavily_search", payload["tools"])

    def test_query_endpoint_returns_structured_response(self) -> None:
        with patch("api.app._run_agent_sync", return_value=FAKE_RESULT):
            response = self.client.post(
                "/agent/query",
                json={"query": "What is RAG in AI?", "max_iterations": 5},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), FAKE_RESULT)

    def test_trace_dedupes_duplicate_finish_steps(self) -> None:
        trace = _extract_trace(
            [
                AIMessage(content="Draft answer."),
                AIMessage(content="Reviewed answer."),
            ],
            final_answer_reviewed=True,
        )

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["action"], "FINISH")

    def test_sources_are_sorted_by_quality(self) -> None:
        sources = _extract_sources(
            [
                ToolMessage(
                    content=(
                        "Top results:\n"
                        "- Blog post - https://exampleblog.com/post - Opinionated summary.\n"
                        "- CDC report - https://www.cdc.gov/report.pdf - Official PDF report.\n"
                        "- Wikipedia - https://en.wikipedia.org/wiki/Retrieval-augmented_generation - Encyclopedia summary."
                    ),
                    tool_call_id="1",
                )
            ]
        )

        self.assertGreaterEqual(len(sources), 3)
        self.assertEqual(sources[0]["url"], "https://www.cdc.gov/report.pdf")
        self.assertEqual(sources[1]["url"], "https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

    def test_stream_endpoint_emits_final_result(self) -> None:
        def fake_stream_run(query, max_iterations, model_name, depth_mode=None, callbacks=None):
            for callback in callbacks or []:
                callback.on_llm_new_token("RAG ")
                callback.on_llm_new_token("answer")
            return FAKE_RESULT

        with patch("api.app._run_agent_sync", side_effect=fake_stream_run):
            with self.client.stream(
                "POST",
                "/agent/stream",
                json={"query": "What is RAG in AI?", "max_iterations": 5},
            ) as response:
                body = response.read().decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "token"', body)
        self.assertIn('"type": "final"', body)
        self.assertIn('"result"', body)
        self.assertIn('"trace"', body)
        self.assertEqual(body.count('"type": "final"'), 1)

    def test_best_available_answer_ignores_stale_draft_after_retry(self) -> None:
        answer = _best_available_answer(
            {
                "final_answer": "Stale draft answer.",
                "final_answer_reviewed": False,
                "messages": [
                    AIMessage(content="Stale draft answer."),
                    SystemMessage(content="Validation failed. Retry with the required structure and use tools if needed."),
                    AIMessage(content=""),
                ],
            }
        )

        self.assertNotEqual(answer, "Stale draft answer.")
        self.assertTrue(answer)


if __name__ == "__main__":
    unittest.main()
