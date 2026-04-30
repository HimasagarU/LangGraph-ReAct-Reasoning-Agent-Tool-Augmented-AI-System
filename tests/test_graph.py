from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph


def _initial_state(query: str, answer_type: str) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "intent": "discovery",
        "answer_type": answer_type,
        "depth_mode": "standard",
        "tool_calls_made": [],
        "iteration_count": 0,
        "max_iterations": 4,
        "confidence": "low",
        "rewritten_query": "",
        "retry_count": 0,
        "needs_retry": False,
        "final_answer": "",
        "final_answer_reviewed": False,
        "validation_errors": [],
        "answer_format_ok": False,
    }


class _FakeModel:
    def __init__(self, agent_responses: list[str], review_responses: list[str] | None = None) -> None:
        self.agent_responses = agent_responses
        self.review_responses = review_responses or agent_responses
        self.agent_calls = 0
        self.review_calls = 0

    def bind_tools(self, tools):  # noqa: ANN001 - langchain-compatible shim
        return self

    def invoke(self, prompt):
        system_text = str(prompt[0].content if isinstance(prompt, list) and prompt else "")
        if "production ReAct agent" in system_text:
            index = min(self.agent_calls, len(self.agent_responses) - 1)
            self.agent_calls += 1
            return AIMessage(content=self.agent_responses[index])
        if "final answer reviewer" in system_text:
            index = min(self.review_calls, len(self.review_responses) - 1)
            self.review_calls += 1
            return AIMessage(content=self.review_responses[index])
        raise AssertionError(f"Unexpected prompt: {system_text}")


class GraphBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        build_graph.cache_clear()

    def tearDown(self) -> None:
        build_graph.cache_clear()

    def test_ambiguous_query_returns_single_clarification_without_model(self) -> None:
        with patch("agent.graph._load_model", side_effect=AssertionError("model should not load")):
            graph = build_graph(model_name="test-model")
            state = graph.invoke(_initial_state("apple", "ambiguous"))

        answer = str(state.get("final_answer") or "")
        self.assertTrue(answer.endswith("?"))
        self.assertEqual(answer.count("?"), 1)
        self.assertEqual(state.get("tool_calls_made"), [])
        self.assertTrue(state.get("final_answer_reviewed"))
        self.assertTrue(state.get("answer_format_ok"))

    def test_calculation_query_uses_direct_fast_path_without_model(self) -> None:
        with patch("agent.graph._load_model", side_effect=AssertionError("model should not load")):
            graph = build_graph(model_name="test-model")
            state = graph.invoke(_initial_state("Calculate 18 * (27 + 5)", "calculation"))

        self.assertIn("576", str(state.get("final_answer") or ""))
        self.assertEqual(state.get("tool_calls_made"), [])
        self.assertTrue(state.get("final_answer_reviewed"))
        self.assertTrue(state.get("answer_format_ok"))

    def test_invalid_comparison_retries_until_table_format_is_produced(self) -> None:
        improved_answer = """1. Definition
RAG retrieves external context before generation, while fine-tuning updates model weights.

2. Intuition
RAG changes what the model can read at inference time, while fine-tuning changes what the model has learned.

3. Table comparison
| Criterion | RAG | Fine-tuning |
| --- | --- | --- |
| Best for | Fast knowledge updates | Stable task specialization |
| Tradeoff | Retrieval quality matters | Training cost is higher |

4. Use cases
RAG works well when knowledge changes often. Fine-tuning is better when style or behavior must stay consistent.

5. Key insights
RAG is flexible, while fine-tuning is deeper but costlier."""

        fake_model = _FakeModel(
            agent_responses=[
                "RAG is different from fine-tuning.",
                improved_answer,
            ],
            review_responses=[
                "RAG is different from fine-tuning.",
                "RAG is different from fine-tuning.",
                improved_answer,
                improved_answer,
            ],
        )

        with patch("agent.graph._load_model", return_value=fake_model):
            graph = build_graph(model_name="test-model")
            state = graph.invoke(_initial_state("RAG vs fine-tuning", "comparison"))

        self.assertEqual(state.get("retry_count"), 1)
        self.assertTrue(state.get("answer_format_ok"))
        self.assertEqual(state.get("validation_errors"), [])
        self.assertEqual(state.get("final_answer"), improved_answer)
        self.assertIn("| Criterion | RAG | Fine-tuning |", str(state.get("final_answer") or ""))


if __name__ == "__main__":
    unittest.main()
