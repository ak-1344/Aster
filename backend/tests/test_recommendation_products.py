"""Tests for behavioral product mapping in recommendation_node."""

from __future__ import annotations

import unittest
from pathlib import Path

from backend.app.nodes.feature_engineering_node import generate_features
from backend.app.nodes.recommendation_node import build_recommendations
from backend.app.nodes.segmentation_node import segment_customers

BEHAVIORAL_PRODUCTS = {
    "debt_consolidation_loan",
    "cash_advance_alternative_credit_line",
    "premium_rewards_card",
    "reactivation_offer",
}


class RecommendationProductTests(unittest.TestCase):
    def test_real_cluster_includes_behavioral_product_tag(self) -> None:
        dataset_path = Path("backend/data/raw/CC GENERAL.csv")
        features, _ = generate_features(dataset_path=dataset_path)
        labels = segment_customers(features=features, n_clusters=3)["labels"]
        recommendations = build_recommendations(features=features, labels=labels)

        all_tags: set[str] = set()
        for cluster_rec in recommendations["cluster_recommendations"]:
            all_tags.update(cluster_rec.get("behavioral_products", []))
            all_tags.update(cluster_rec.get("cross_sell", []))

        matched = all_tags & BEHAVIORAL_PRODUCTS
        self.assertGreater(
            len(matched),
            0,
            msg=f"Expected at least one behavioral product tag, got {all_tags}",
        )

    def test_tier_assignment_includes_priority_and_dormant(self) -> None:
        """Regression: dataset-wide thresholds must yield priority and dormant tiers."""
        dataset_path = Path("backend/data/raw/CC GENERAL.csv")
        features, _ = generate_features(dataset_path=dataset_path)
        # n_clusters=17 isolates low/high spend cohorts on CC GENERAL; n=3 leaves all
        # cluster medians above the dataset q33 floor so dormant is unreachable.
        labels = segment_customers(features=features, n_clusters=17)["labels"]
        recommendations = build_recommendations(features=features, labels=labels)

        tiers = set(recommendations["cluster_tiers"].values())
        self.assertIn("priority", tiers, msg=f"Expected a priority cluster, got tiers={tiers}")
        self.assertIn("dormant", tiers, msg=f"Expected a dormant cluster, got tiers={tiers}")
        self.assertNotEqual(tiers, {"regular"}, msg="All-regular tier collapse should not occur")


if __name__ == "__main__":
    unittest.main()
