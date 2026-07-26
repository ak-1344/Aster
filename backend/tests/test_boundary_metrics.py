"""Tests for customer boundary metrics and borderline explanations."""

from __future__ import annotations

import unittest
from pathlib import Path

from backend.app.decision_engine.decision_engine import (
    _append_boundary_fields,
    _BOUNDARY_RATIO_THRESHOLD,
    generate_explanations,
)
from backend.app.nodes.feature_engineering_node import generate_features
from backend.app.nodes.segmentation_node import segment_customers


class BoundaryMetricTests(unittest.TestCase):
    def test_segmentation_exposes_boundary_distance_ratio(self) -> None:
        features, _ = generate_features(dataset_path=Path("backend/data/raw/CC GENERAL.csv"))
        result = segment_customers(features=features, n_clusters=3)
        metrics = result.get("customer_boundary_metrics", [])

        self.assertGreater(len(metrics), 0)
        for metric in metrics[:10]:
            self.assertIn("boundary_distance_ratio", metric)
            self.assertIn("nearest_alternate_cluster", metric)
            ratio = metric["boundary_distance_ratio"]
            self.assertIsInstance(ratio, (int, float))
            self.assertGreaterEqual(float(ratio), 0.0)

    def test_borderline_sentence_only_above_threshold(self) -> None:
        base = {
            "customer_id": "C100",
            "cluster_label": 1,
            "explanation": "Assigned to Segment 1 primarily due to high monthly spend.",
        }
        below = _append_boundary_fields(
            dict(base),
            {
                "customer_id": "C100",
                "cluster_label": 1,
                "nearest_alternate_cluster": 2,
                "boundary_distance_ratio": 0.5,
            },
        )
        above = _append_boundary_fields(
            dict(base),
            {
                "customer_id": "C100",
                "cluster_label": 1,
                "nearest_alternate_cluster": 2,
                "boundary_distance_ratio": 0.9,
            },
        )

        self.assertNotIn("borderline between Segment", below["explanation"])
        self.assertIn("borderline between Segment 1 and Segment 2", above["explanation"])
        self.assertGreater(_BOUNDARY_RATIO_THRESHOLD, 0.5)
        self.assertLess(_BOUNDARY_RATIO_THRESHOLD, 0.9)

    def test_explanations_include_boundary_fields_from_segmentation(self) -> None:
        features, saved_path = generate_features(
            dataset_path=Path("backend/data/raw/CC GENERAL.csv")
        )
        segmentation = segment_customers(features=features, n_clusters=3)
        node_outputs = {
            "feature_engineering": {"features_path": str(saved_path)},
            "segmentation": segmentation,
        }

        explanations = generate_explanations(node_outputs)
        customer_rows = explanations.get("customer_explanations", [])
        self.assertGreater(len(customer_rows), 0)

        with_ratio = [row for row in customer_rows if "boundary_distance_ratio" in row]
        self.assertGreater(len(with_ratio), 0)


if __name__ == "__main__":
    unittest.main()
