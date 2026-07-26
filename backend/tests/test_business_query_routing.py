"""Regression tests for business-phrased query intent routing."""

from __future__ import annotations

import unittest

from starlette.testclient import TestClient

from backend.app.api.main import app


class BusinessQueryRoutingTests(unittest.TestCase):
    """Ensure analyst-style queries route through segmentation and recommendation."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def _assert_segmentation_recommendation_workflow(self, query: str) -> None:
        response = self.client.post("/query", json={"query": query})

        self.assertEqual(response.status_code, 200, msg=response.text)
        body = response.json()
        executed = body["metadata"]["nodes_executed"]

        self.assertIn("segmentation", executed, msg=f"query={query!r} nodes={executed}")
        self.assertIn("recommendation", executed, msg=f"query={query!r} nodes={executed}")

    def test_investment_products_query_routes_to_segmentation(self) -> None:
        self._assert_segmentation_recommendation_workflow(
            "Find customers suitable for investment products"
        )

    def test_inactive_customers_query_routes_to_segmentation(self) -> None:
        self._assert_segmentation_recommendation_workflow(
            "Which customers are becoming inactive?"
        )

    def test_retention_campaign_query_routes_to_segmentation(self) -> None:
        self._assert_segmentation_recommendation_workflow(
            "Which customers should be targeted for retention campaigns?"
        )

    def test_literal_segmentation_query_still_routes(self) -> None:
        self._assert_segmentation_recommendation_workflow(
            "Segment the customers into groups"
        )


if __name__ == "__main__":
    unittest.main()
