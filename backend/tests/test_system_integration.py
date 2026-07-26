"""System integration tests for intent routing, clustering, and answer formatting."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from starlette.testclient import TestClient

from backend.app.api.main import app
from backend.app.context_builder.context_builder import build_context, infer_intent
from backend.app.planner.planner import (
    build_execution_plan,
    extract_customer_id,
    format_high_impact_answer,
    select_explanation_tool,
    TOOL_CUSTOMER_LOOKUP,
)
from backend.app.query_manager.query_manager import route_intent
from backend.app.tools.segmentation import run_segmentation_pipeline


class IntentRoutingIntegrationTests(unittest.TestCase):
    def test_high_end_credit_card_routes_to_segmentation(self) -> None:
        query = "Who should I market our high-end credit card to?"
        context = build_context(query)

        self.assertEqual(context["intent"], "segmentation")
        self.assertEqual(route_intent(query), "segmentation")
        self.assertEqual(infer_intent(context["normalized_query"]), "segmentation")

    def test_cash_advance_reliance_routes_to_segmentation(self) -> None:
        query = "Which people are relying heavily on cash advances?"
        context = build_context(query)

        self.assertEqual(context["intent"], "segmentation")
        self.assertEqual(route_intent(query), "segmentation")

    def test_descriptive_statistics_routes_to_descriptive(self) -> None:
        query = "Show descriptive statistics for the dataset"
        context = build_context(query)

        self.assertEqual(context["intent"], "descriptive")
        self.assertEqual(route_intent(query), "descriptive")

    def test_customer_segment_why_routes_to_explanation(self) -> None:
        query = "Why was customer C10002 placed in this segment?"
        context = build_context(query)

        self.assertEqual(context["intent"], "explanation")
        self.assertEqual(route_intent(query), "explanation")
        self.assertEqual(extract_customer_id(query), "C10002")

    def test_suspicious_customer_triggers_entity_lookup_explanation(self) -> None:
        query = "Is customer C10005 suspicious?"
        context = build_context(query)

        self.assertEqual(context["intent"], "explanation")
        self.assertEqual(route_intent(query), "explanation")
        self.assertEqual(extract_customer_id(query), "C10005")
        self.assertEqual(select_explanation_tool(query), TOOL_CUSTOMER_LOOKUP)


class SystemOutputIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = run_segmentation_pipeline()

    def test_segmentation_pipeline_returns_personas_and_recommendations(self) -> None:
        summary = self.pipeline["persona_summary"]
        frame = self.pipeline["dataframe"]

        self.assertTrue(summary)
        self.assertGreater(len(frame), 0)
        self.assertIn("persona", frame.columns)
        self.assertIn("recommended_product", frame.columns)

        for row in summary:
            self.assertIn("persona", row)
            self.assertIn("recommended_product", row)
            self.assertGreater(row["customer_count"], 0)
            self.assertTrue(str(row["recommended_product"]).strip())

        personas = {row["persona"] for row in summary}
        self.assertTrue(personas)
        self.assertTrue(frame["recommended_product"].astype(str).str.len().gt(0).all())

    def test_segmentation_answer_contains_markdown_cust_id_table(self) -> None:
        query = "Who should I market our high-end credit card to?"
        context = build_context(query)
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "INVALID_KEY_FOR_TESTING"}):
            plan = build_execution_plan(context)

        answer = format_high_impact_answer(context, plan, node_outputs={})
        markdown = answer["markdown"]

        self.assertIn("| CUST_ID |", markdown)
        self.assertIn("CREDIT_UTILIZATION", markdown)
        self.assertIn("Target Customer Table", markdown)
        self.assertTrue(answer["target_customers"])
        self.assertIn("CUST_ID", answer["target_customers"][0])

    def test_api_response_includes_planning_path_and_routing_reason(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/query",
            json={"query": "Who should I market our high-end credit card to? integration"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        metadata = body["metadata"]

        self.assertIn(metadata["planning_path"], {"llm_reasoned", "rule_based_fallback"})
        self.assertIn("routing_reason", metadata)
        self.assertIsNotNone(metadata["routing_reason"])
        self.assertIn("similarity=", metadata["routing_reason"])

        markdown = body.get("answer_markdown") or body.get("agent_answer", {}).get("markdown", "")
        self.assertIn("CUST_ID", markdown)


if __name__ == "__main__":
    unittest.main()
