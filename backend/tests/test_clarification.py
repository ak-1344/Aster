"""Tests for human-in-the-loop clarification routing."""

from __future__ import annotations

import unittest
import uuid

from starlette.testclient import TestClient

from backend.app.api.main import app
from backend.app.context_builder.context_builder import build_context, detect_ambiguity


class ClarificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_unsupported_filter_only_query_triggers_clarification(self) -> None:
        query = "list customers in Chennai"
        context = build_context(query)
        clarification = detect_ambiguity(context, context["dataset_path"])

        self.assertIsNotNone(clarification)
        self.assertIn("question", clarification)

        response = self.client.post("/query", json={"query": query})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "clarification_needed")
        self.assertEqual(body["original_query"], query)
        self.assertIn("city/location", body["question"])

    def test_business_phrased_query_does_not_trigger_clarification(self) -> None:
        query = "Find customers suitable for investment products"
        context = build_context(query)

        self.assertIsNone(detect_ambiguity(context, context["dataset_path"]))

        response = self.client.post("/query", json={"query": query})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotEqual(body.get("status"), "clarification_needed")
        executed = body["metadata"]["nodes_executed"]
        self.assertIn("segmentation", executed)
        self.assertIn("recommendation", executed)

    def test_clarification_round_trip_returns_segmentation_output(self) -> None:
        original_query = f"list customers in Chennai issue008 {uuid.uuid4().hex[:8]}"
        first = self.client.post("/query", json={"query": original_query})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "clarification_needed")

        second = self.client.post(
            "/query",
            json={
                "query": original_query,
                "clarification_response": "segment customers into 3 clusters",
            },
        )
        self.assertEqual(second.status_code, 200)
        body = second.json()
        self.assertNotEqual(body.get("status"), "clarification_needed")
        self.assertEqual(body["workflow_name"], "segmentation_workflow")
        executed = body["metadata"]["nodes_executed"]
        self.assertIn("segmentation", executed)
        self.assertIn("recommendation", executed)


if __name__ == "__main__":
    unittest.main()
