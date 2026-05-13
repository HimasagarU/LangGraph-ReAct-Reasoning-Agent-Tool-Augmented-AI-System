from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from agent import tools


class ToolTests(unittest.TestCase):
    def test_tavily_fallback_to_wikipedia_on_empty_results(self) -> None:
        fallback_result = {
            "title": "Test Title",
            "url": "https://en.wikipedia.org/wiki/Test_Title",
            "snippet": "Test summary",
            "score": 3,
        }
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test"}):
            with patch("agent.tools.TavilyClient") as mock_client:
                mock_client.return_value.search.return_value = {"results": []}
                with patch("agent.tools._build_wikipedia_fallback_result", return_value=fallback_result):
                    payload = tools.tavily_search.invoke({"query": "test"})

        data = json.loads(payload)
        self.assertEqual(data.get("fallback"), "wikipedia_lookup")
        self.assertEqual(len(data.get("results", [])), 1)
        self.assertEqual(data["results"][0]["url"], fallback_result["url"])

    def test_page_fetch_extracts_title_and_content(self) -> None:
        class _FakeResponse:
            def __init__(self, text: str) -> None:
                self.text = text

            def raise_for_status(self) -> None:
                return None

        html = (
            "<html><head><title>Example Page</title></head><body>"
            "<p>Hello world</p><p>More content for length padding.</p>"
            "<p>Even more content to exceed the minimum length.</p>"
            "</body></html>"
        )

        with patch("agent.tools.requests.get", return_value=_FakeResponse(html)):
            payload = tools.page_fetch.invoke({"url": "https://example.com"})

        data = json.loads(payload)
        self.assertTrue(data.get("results"))
        self.assertEqual(data["results"][0]["title"], "Example Page")
        self.assertIn("Hello world", data["results"][0]["snippet"])


if __name__ == "__main__":
    unittest.main()
