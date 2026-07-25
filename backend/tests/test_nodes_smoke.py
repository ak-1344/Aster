from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.app.nodes.analytics_node import build_descriptive_statistics
from backend.app.nodes.eda_node import build_exploratory_summary
from backend.app.nodes.evaluation_node import evaluate_segmentation
from backend.app.nodes.feature_engineering_node import generate_features
from backend.app.nodes.recommendation_node import build_recommendations
from backend.app.nodes.segmentation_node import segment_customers
from backend.app.nodes.visualization_node import build_visualization_payload


class NodeSmokeTests(unittest.TestCase):
    def test_analytics_node_returns_summary(self) -> None:
        summary = build_descriptive_statistics(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        self.assertGreater(summary["row_count"], 0)
        self.assertIn("BALANCE", summary["columns"])
        self.assertIn("mean", summary["numeric_summary"]["BALANCE"])

    def test_eda_node_returns_exploratory_summary(self) -> None:
        summary = build_exploratory_summary(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        self.assertGreater(summary["row_count"], 0)
        self.assertIn("CUST_ID", summary["missing_values"])
        self.assertGreaterEqual(summary["missing_percentage"]["BALANCE"], 0.0)

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
        result = segment_customers(features=features, n_clusters=2)
        self.assertIn("labels", result)
        self.assertEqual(len(result["labels"]), len(features))

    def test_recommendation_node_returns_actions(self) -> None:
        features, _ = generate_features(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        recommendations = build_recommendations(features=features)
        self.assertGreater(len(recommendations), 0)
        self.assertIn("customer_id", recommendations[0])

    def test_evaluation_node_returns_metrics(self) -> None:
        features, _ = generate_features(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        labels = segment_customers(features=features, n_clusters=2)["labels"]
        evaluation = evaluate_segmentation(features=features, labels=labels)
        self.assertIn("silhouette_score", evaluation)
        self.assertIn("cluster_sizes", evaluation)

    def test_visualization_node_returns_payload(self) -> None:
        features, _ = generate_features(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        labels = segment_customers(features=features, n_clusters=2)["labels"]
        payload = build_visualization_payload(features=features, labels=labels)
        self.assertEqual(payload["chart_type"], "scatter")
        self.assertIn("data", payload)


if __name__ == "__main__":
    unittest.main()
