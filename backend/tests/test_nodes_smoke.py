from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.app.nodes.analytics_node import build_descriptive_statistics
from backend.app.context_builder.context_builder import build_context, normalize_query
from backend.app.nodes.eda_node import build_exploratory_summary
from backend.app.nodes.evaluation_node import evaluate_segmentation
from backend.app.nodes.feature_engineering_node import generate_features
from backend.app.planner.planner import build_execution_plan
from backend.app.query_manager.query_manager import orchestrate_query, route_intent
from backend.app.nodes.recommendation_node import build_recommendations
from backend.app.nodes.segmentation_node import segment_customers
from backend.app.nodes.visualization_node import build_visualization_payload


class NodeSmokeTests(unittest.TestCase):
    def test_analytics_node_returns_summary(self) -> None:
        summary = build_descriptive_statistics(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        self.assertGreater(summary["row_count"], 0)
        self.assertIn("BALANCE", summary["columns"])
        self.assertIn("mean", summary["numeric_summary"]["BALANCE"])
        self.assertIn("missing_values", summary)

    def test_eda_node_returns_exploratory_summary(self) -> None:
        summary = build_exploratory_summary(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        self.assertGreater(summary["row_count"], 0)
        self.assertIn("CUST_ID", summary["missing_values"])
        self.assertGreaterEqual(summary["missing_percentage"]["BALANCE"], 0.0)
        self.assertIn("numeric_correlations", summary)

    def test_feature_engineering_node_writes_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "customer_features.csv"
            features, saved_path = generate_features(
                dataset_path=Path("backend/data/raw/CC GENERAL.csv"),
                output_path=output_path,
            )
            self.assertTrue(saved_path.exists())
            self.assertGreater(len(features), 0)
            self.assertIn("monthly_spend", features.columns)

    def test_segmentation_node_returns_labels(self) -> None:
        features, _ = generate_features(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        result = segment_customers(features=features, n_clusters=3)
        self.assertIn("labels", result)
        self.assertEqual(len(result["labels"]), len(features))
        self.assertIn("clustered_customers", result)

    def test_recommendation_node_returns_actions(self) -> None:
        features, _ = generate_features(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        labels = segment_customers(features=features, n_clusters=3)["labels"]
        recommendations = build_recommendations(features=features, labels=labels)
        self.assertGreater(len(recommendations["cluster_recommendations"]), 0)
        self.assertIn("customer_recommendations", recommendations)

    def test_evaluation_node_returns_metrics(self) -> None:
        features, _ = generate_features(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        labels = segment_customers(features=features, n_clusters=3)["labels"]
        evaluation = evaluate_segmentation(features=features, labels=labels)
        self.assertIn("silhouette_score", evaluation)
        self.assertIn("cluster_sizes", evaluation)
        self.assertIn("cluster_count", evaluation)

    def test_visualization_node_returns_payload(self) -> None:
        features, _ = generate_features(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        labels = segment_customers(features=features, n_clusters=3)["labels"]
        payload = build_visualization_payload(features=features, labels=labels)
        self.assertIn("scatter", payload)
        self.assertIn("cluster_size_bar", payload)

    def test_context_builder_routes_segmentation_query(self) -> None:
        context = build_context("segment customers into 3 clusters")
        self.assertEqual(context["intent"], "segmentation")
        self.assertIn("customer_clusters", context["entities"])
        self.assertEqual(context["filters"]["n_clusters"], 3)

    def test_query_manager_orchestrates_descriptive_query(self) -> None:
        result = orchestrate_query("show descriptive statistics for the dataset")
        self.assertTrue(result["executable"])
        self.assertEqual(result["intent"], "descriptive")
        self.assertEqual(result["plan"]["entrypoint"], "analytics")

    def test_planner_builds_segmentation_workflow(self) -> None:
        context = build_context("segment customers into 4 clusters")
        plan = build_execution_plan(context)
        self.assertTrue(plan["executable"])
        self.assertEqual(plan["workflow_name"], "segmentation_workflow")
        self.assertEqual(plan["entrypoint"], "feature_engineering")
        self.assertEqual(plan["exitpoint"], "visualization")
        self.assertEqual(plan["step_count"], 5)

    def test_query_normalization_and_intent_routing(self) -> None:
        self.assertEqual(normalize_query("  Segment! Customers??  "), "segment customers")
        self.assertEqual(route_intent("Give me a descriptive summary"), "descriptive")


if __name__ == "__main__":
    unittest.main()
