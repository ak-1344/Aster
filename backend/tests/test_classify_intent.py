"""Regression tests for planner classify_intent token matching."""

from __future__ import annotations

import unittest

from starlette.testclient import TestClient

from backend.app.api.main import app
from backend.app.planner.planner import classify_intent
from backend.app.context_builder.context_builder import build_context


class ClassifyIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_show_premium_customers_is_not_explanation_only(self) -> None:
        query = "Show premium customers from Chennai."
        context = build_context(query)

        self.assertNotEqual(classify_intent(context), "explanation_only")

        response = self.client.post("/query", json={"query": query})
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertNotEqual(body.get("intent_classification"), "explanation_only")
        executed = body.get("metadata", {}).get("nodes_executed", [])
        self.assertGreater(len(executed), 0)
        self.assertIn("segmentation", executed)

    def test_genuine_explanation_query_still_classifies_explanation_only(self) -> None:
        query = "explain why this customer is in this segment"
        context = build_context(query)

        self.assertEqual(classify_intent(context), "explanation_only")

        response = self.client.post("/query", json={"query": query})
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body.get("intent_classification"), "explanation_only")
        self.assertEqual(body.get("metadata", {}).get("nodes_executed"), [])


if __name__ == "__main__":
    unittest.main()
