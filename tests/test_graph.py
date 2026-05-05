from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph


def _initial_state(query: str, answer_type: str, reasoning_budget: str = "medium") -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "intent": "discovery",
        "answer_type": answer_type,
        "reasoning_budget": reasoning_budget,
        "tool_calls_made": [],
        "iteration_count": 0,
        "max_iterations": 4,
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
        if "production ReAct assistant" in system_text:
            index = min(self.agent_calls, len(self.agent_responses) - 1)
            self.agent_calls += 1
            return AIMessage(content=self.agent_responses[index])
        if "final answer reviewer" in system_text or "strict final-answer formatter" in system_text:
            index = min(self.review_calls, len(self.review_responses) - 1)
            self.review_calls += 1
            return AIMessage(content=self.review_responses[index])
        raise AssertionError(f"Unexpected prompt: {system_text[:200]}")


class GraphBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        build_graph.cache_clear()

    def tearDown(self) -> None:
        build_graph.cache_clear()

    def test_ambiguous_query_returns_single_clarification_without_model(self) -> None:
        with patch("agent.graph._load_model", side_effect=AssertionError("model should not load")):
            graph = build_graph(model_name="test-model")
            state = graph.invoke(_initial_state("apple", "ambiguous", "shallow"))

        answer = str(state.get("final_answer") or "")
        self.assertTrue(answer.endswith("?"))
        self.assertEqual(answer.count("?"), 1)
        self.assertEqual(state.get("tool_calls_made"), [])
        self.assertTrue(state.get("final_answer_reviewed"))
        self.assertTrue(state.get("answer_format_ok"))

    def test_calculation_query_uses_direct_fast_path_without_model(self) -> None:
        with patch("agent.graph._load_model", side_effect=AssertionError("model should not load")):
            graph = build_graph(model_name="test-model")
            state = graph.invoke(_initial_state("Calculate 18 * (27 + 5)", "calculation", "shallow"))

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
            state = graph.invoke(_initial_state("RAG vs fine-tuning", "comparison", "medium"))

        self.assertEqual(state.get("retry_count"), 1)
        self.assertTrue(state.get("answer_format_ok"))
        self.assertEqual(state.get("validation_errors"), [])
        self.assertEqual(state.get("final_answer"), improved_answer)
        self.assertIn("| Criterion | RAG | Fine-tuning |", str(state.get("final_answer") or ""))

    def test_retry_loop_does_not_return_stale_draft_answer(self) -> None:
        fake_model = _FakeModel(
            agent_responses=[
                "RAG is different from fine-tuning.",
                "",
            ],
            review_responses=[
                "RAG is different from fine-tuning.",
                "RAG is different from fine-tuning.",
            ],
        )

        with patch("agent.graph._load_model", return_value=fake_model):
            graph = build_graph(model_name="test-model")
            state = graph.invoke(_initial_state("RAG vs fine-tuning", "comparison", "medium"))

        self.assertNotEqual(state.get("final_answer"), "RAG is different from fine-tuning.")
        self.assertNotEqual(str(state.get("final_answer") or ""), "RAG is different from fine-tuning.")

    def test_fact_query_produces_answer_through_review(self) -> None:
        """Fact query goes through review node which reformats the answer."""
        fake_model = _FakeModel(
            agent_responses=[
                "The prize is one million dollars.",
                "The prize is one million dollars.",
            ],
            review_responses=[
                "**Answer:** The Turing Award cash prize is $1 million.\n\n**Sources:** Tool results",
                "**Answer:** The Turing Award cash prize is $1 million.\n\n**Sources:** Tool results",
            ],
        )

        with patch("agent.graph._load_model", return_value=fake_model):
            graph = build_graph(model_name="test-model")
            state = graph.invoke(_initial_state("What is the cash prize of the Turing Award?", "fact", "medium"))

        answer = str(state.get("final_answer") or "")
        self.assertTrue(len(answer) > 0, "Fact query should produce a non-empty answer")
        self.assertTrue(state.get("final_answer_reviewed"))


class ReasoningBudgetTests(unittest.TestCase):
    """Tests for the adaptive reasoning budget computation."""

    def setUp(self) -> None:
        build_graph.cache_clear()

    def tearDown(self) -> None:
        build_graph.cache_clear()

    def test_fact_query_uses_shallow_budget(self) -> None:
        """Fact queries should auto-compute to shallow budget."""
        from agent.graph import _compute_reasoning_budget
        budget = _compute_reasoning_budget("fact", 1.0, "What is the cash prize of the Turing Award?")
        self.assertEqual(budget, "shallow")

    def test_ambiguous_query_stays_shallow(self) -> None:
        """Ambiguous queries should use shallow budget and not expand."""
        with patch("agent.graph._load_model", side_effect=AssertionError("model should not load")):
            graph = build_graph(model_name="test-model")
            state_input = _initial_state("apple", "ambiguous")
            state_input["reasoning_budget"] = ""
            state_input["classifier_confidence"] = 0.0
            state = graph.invoke(state_input)

        budget = str(state.get("reasoning_budget") or state.get("metrics", {}).get("reasoning_budget", ""))
        self.assertEqual(budget, "shallow")
        self.assertEqual(state.get("rewrite_variants", []), [])

    def test_explanation_query_can_use_deep_budget(self) -> None:
        """Complex explanation queries with low confidence should get deep budget."""
        from agent.graph import _compute_reasoning_budget
        budget = _compute_reasoning_budget("explanation", 0.4, "How do transformers handle long-range dependencies using self-attention mechanisms?")
        self.assertEqual(budget, "deep")

    def test_simple_explanation_uses_medium_budget(self) -> None:
        """Short explanation queries with high confidence should get medium budget."""
        from agent.graph import _compute_reasoning_budget
        budget = _compute_reasoning_budget("explanation", 0.9, "What is RAG?")
        self.assertEqual(budget, "medium")


class CriticTests(unittest.TestCase):
    """Tests for the content-aware critic checks."""

    def test_critic_catches_overclaim(self) -> None:
        from agent.graph import _critic_content_checks
        issues = _critic_content_checks(
            "Transformers always outperform RNNs in every single task.",
            "explanation",
            "explanatory",
        )
        overclaim_issues = [i for i in issues if "overclaim" in i]
        self.assertTrue(len(overclaim_issues) > 0, f"Expected overclaim issue, got: {issues}")

    def test_critic_ignores_hedged_language(self) -> None:
        from agent.graph import _critic_content_checks
        issues = _critic_content_checks(
            "Transformers generally outperform RNNs, but this is not always the case.",
            "explanation",
            "explanatory",
        )
        overclaim_issues = [i for i in issues if "overclaim" in i]
        self.assertEqual(len(overclaim_issues), 0, f"Should not flag hedged language, got: {issues}")

    def test_critic_catches_taxonomy_confusion(self) -> None:
        from agent.graph import _critic_content_checks
        issues = _critic_content_checks(
            "Agentic AI is AI agent that operates autonomously.",
            "explanation",
            "explanatory",
        )
        taxonomy_issues = [i for i in issues if "taxonomy" in i]
        self.assertTrue(len(taxonomy_issues) > 0, f"Expected taxonomy issue, got: {issues}")


if __name__ == "__main__":
    unittest.main()
