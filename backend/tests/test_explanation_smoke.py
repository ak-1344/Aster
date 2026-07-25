"""Explainability smoke tests for ASTER."""

from __future__ import annotations

import unittest

from starlette.testclient import TestClient

from backend.app.api.main import app
from backend.app.decision_engine.decision_engine import generate_explanations
from backend.app.query_manager.query_manager import execute_query


class ExplainabilitySmokeTests(unittest.TestCase):
    def test_rule_based_explanations_are_non_empty_for_segmentation(self) -> None:
        """Segmentation /query responses include per-customer explanations."""
        client = TestClient(app)
        response = client.post(
            "/query",
            json={"query": "segment customers into 3 clusters"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        explanations = body.get("explanations", {})
        customer_explanations = explanations.get("customer_explanations", [])
        segment_summaries = explanations.get("segment_summaries", [])

        self.assertGreater(len(customer_explanations), 0)
        self.assertTrue(all(item.get("explanation") for item in customer_explanations))
        self.assertGreater(len(segment_summaries), 0)

    def test_generate_explanations_returns_empty_for_descriptive_only(self) -> None:
        """Non-segmentation workflows do not emit customer cluster explanations."""
        descriptive = execute_query("show descriptive statistics for the dataset")
        self.assertEqual(descriptive["workflow_name"], "descriptive_workflow")
        self.assertEqual(descriptive["explanations"]["customer_explanations"], [])


if __name__ == "__main__":
    unittest.main()
