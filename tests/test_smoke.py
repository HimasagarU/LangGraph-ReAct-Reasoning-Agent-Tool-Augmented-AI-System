import unittest

from agent.intent import (
    ANSWER_AMBIGUOUS,
    ANSWER_CALCULATION,
    ANSWER_COMPARISON,
    ANSWER_EXPLANATION,
    ANSWER_FACT,
    INTENT_COMPARATIVE,
    INTENT_DISCOVERY,
    INTENT_EXPLANATORY,
    classify_answer_type,
    classify_query_intent,
    contains_math_expression,
    preferred_tools_for_intent,
)


class IntentSmokeTests(unittest.TestCase):
    def test_explanatory_query(self) -> None:
        self.assertEqual(classify_query_intent("What is RAG in AI?"), INTENT_EXPLANATORY)

    def test_comparative_query(self) -> None:
        self.assertEqual(classify_query_intent("compare RAG vs fine-tuning"), INTENT_COMPARATIVE)

    def test_preferred_tool_order(self) -> None:
        self.assertEqual(preferred_tools_for_intent("technical")[0], "calculator")

    def test_page_fetch_in_preferred_tools(self) -> None:
        """Verify page_fetch is preferred for explanation, comparison, and discovery."""
        self.assertIn("page_fetch", preferred_tools_for_intent("explanatory"))
        self.assertIn("page_fetch", preferred_tools_for_intent("comparative"))
        self.assertIn("page_fetch", preferred_tools_for_intent("discovery"))
        # page_fetch is available in technical but not the first choice (calculator is)
        self.assertEqual(preferred_tools_for_intent("technical")[0], "calculator")

    def test_short_question_mark_query_does_not_auto_become_explanatory(self) -> None:
        self.assertEqual(classify_query_intent("Prize?"), INTENT_DISCOVERY)

    def test_fact_answer_type_detection(self) -> None:
        self.assertEqual(classify_answer_type("What is the cash prize of the Turing Award?"), ANSWER_FACT)

    def test_comparison_answer_type_detection(self) -> None:
        self.assertEqual(classify_answer_type("RAG vs fine-tuning"), ANSWER_COMPARISON)

    def test_ambiguous_answer_type_detection(self) -> None:
        self.assertEqual(classify_answer_type("apple"), ANSWER_AMBIGUOUS)

    def test_math_detection_accepts_explicit_expression(self) -> None:
        self.assertTrue(contains_math_expression("Calculate 18 * (27 + 5)"))
        self.assertEqual(classify_answer_type("Calculate 18 * (27 + 5)"), ANSWER_CALCULATION)

    def test_math_detection_rejects_prose_with_symbol_characters(self) -> None:
        self.assertFalse(contains_math_expression("Explain RAG / fine-tuning trade-offs"))
        self.assertEqual(classify_answer_type("Explain RAG / fine-tuning trade-offs"), ANSWER_EXPLANATION)


if __name__ == "__main__":
    unittest.main()
