from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent.observability import build_trace_config, ensure_langsmith_env, langsmith_tracing_enabled


class ObservabilityTests(unittest.TestCase):
    def test_langsmith_enabled_by_api_key(self) -> None:
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}, clear=True):
            self.assertTrue(langsmith_tracing_enabled())

    def test_langsmith_explicit_disable_wins(self) -> None:
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}, clear=True):
            self.assertFalse(langsmith_tracing_enabled())

    def test_ensure_langsmith_env_sets_legacy_flags(self) -> None:
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key", "LANGSMITH_PROJECT": "demo"}, clear=True):
            ensure_langsmith_env()
            self.assertEqual(os.getenv("LANGSMITH_TRACING"), "true")
            self.assertEqual(os.getenv("LANGCHAIN_TRACING_V2"), "true")
            self.assertEqual(os.getenv("LANGCHAIN_PROJECT"), "demo")

    def test_build_trace_config_contains_expected_metadata(self) -> None:
        config = build_trace_config(
            query="What is RAG?",
            model_name="llama-3.3-70b-versatile",
            max_iterations=5,
            reasoning_budget="medium",
            is_streaming=False,
        )
        self.assertEqual(config["run_name"], "langgraph_agent_query")
        self.assertIn("langgraph-react-agent", config["tags"])
        self.assertEqual(config["metadata"]["model_name"], "llama-3.3-70b-versatile")
        self.assertEqual(config["metadata"]["reasoning_budget_requested"], "medium")


if __name__ == "__main__":
    unittest.main()
