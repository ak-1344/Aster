"""API smoke tests for ASTER."""

from __future__ import annotations

import unittest

from starlette.testclient import TestClient

from backend.app.api.main import app


class ApiSmokeTests(unittest.TestCase):
    def test_post_query_segmentation_workflow(self) -> None:
        """POST /query runs the real pipeline against the CC GENERAL dataset."""
        client = TestClient(app)
        response = client.post(
            "/query",
            json={"query": "segment customers into 3 clusters"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["workflow_name"], "segmentation_workflow")
        self.assertEqual(body["intent"], "segmentation")
        self.assertIn("summary", body)
        self.assertIn("statistics", body)
        self.assertIn("recommendations", body)
        self.assertIn("visualizations", body)
        self.assertIn("metadata", body)
        self.assertIn("segmentation", body["metadata"]["nodes_executed"])

    def test_post_query_descriptive_workflow(self) -> None:
        """POST /query runs descriptive/EDA workflow without segmentation nodes."""
        client = TestClient(app)
        response = client.post(
            "/query",
            json={"query": "show descriptive statistics for the dataset"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["workflow_name"], "descriptive_workflow")
        self.assertEqual(body["intent"], "descriptive")
        self.assertIn("statistics", body)
        self.assertIn("descriptive", body["statistics"])
        self.assertIn("exploratory", body["statistics"])
        executed = body["metadata"]["nodes_executed"]
        self.assertIn("analytics", executed)
        self.assertIn("eda", executed)
        self.assertNotIn("segmentation", executed)


if __name__ == "__main__":
    unittest.main()
