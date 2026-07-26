"""Tests for TF-IDF semantic intent classification."""

from __future__ import annotations

import unittest

from starlette.testclient import TestClient

from backend.app.api.main import app
from backend.app.context_builder.context_builder import (
    _SEMANTIC_SIMILARITY_THRESHOLD,
    build_context,
    classify_intent_semantically,
    infer_intent,
)


class SemanticIntentTests(unittest.TestCase):
    def test_business_query_classifies_as_segmentation(self) -> None:
        query = "Find customers suitable for investment products"
        semantic = classify_intent_semantically(query)

        self.assertEqual(semantic["category"], "segmentation")
        self.assertGreaterEqual(semantic["similarity_score"], _SEMANTIC_SIMILARITY_THRESHOLD)
        self.assertEqual(infer_intent(build_context(query)["normalized_query"], semantic=semantic), "segmentation")

    def test_descriptive_query_classifies_correctly(self) -> None:
        query = "show descriptive statistics for the dataset"
        semantic = classify_intent_semantically(query)

        self.assertEqual(semantic["category"], "descriptive")
        self.assertEqual(infer_intent(query, semantic=semantic), "descriptive")

    def test_explanation_query_classifies_correctly(self) -> None:
        query = "explain why this customer is in this segment"
        semantic = classify_intent_semantically(query)

        self.assertEqual(semantic["category"], "explanation")
        self.assertGreaterEqual(semantic["similarity_score"], _SEMANTIC_SIMILARITY_THRESHOLD)
        self.assertEqual(infer_intent(query, semantic=semantic), "explanation")

    def test_filter_only_location_query_lacks_intent_signal(self) -> None:
        """Generic 'customers' overlap must not bypass unsupported-filter clarification."""

        from backend.app.context_builder.context_builder import _has_intent_signal

        query = "list customers in Chennai"
        semantic = classify_intent_semantically(query)

        self.assertEqual(semantic["category"], "segmentation")
        self.assertFalse(_has_intent_signal(query))
        self.assertEqual(infer_intent(query, semantic=semantic), "segmentation")

    def test_api_response_includes_planning_path_and_routing_reason(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/query",
            json={"query": "segment customers into 3 clusters"},
        )

        self.assertEqual(response.status_code, 200)
        metadata = response.json()["metadata"]
        self.assertIn(metadata["planning_path"], {"llm_reasoned", "rule_based_fallback"})
        self.assertIn("routing_reason", metadata)
        self.assertIn("similarity=", metadata["routing_reason"])
        self.assertIn("Matched example", metadata["routing_reason"])


if __name__ == "__main__":
    unittest.main()
