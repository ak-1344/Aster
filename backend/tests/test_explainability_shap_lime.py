"""Explainability tests for SHAP, LIME, and rule-based fallback paths."""

from __future__ import annotations

import os
import unittest

import numpy as np
import pandas as pd

from backend.app.decision_engine.decision_engine import (
    explain_with_lime,
    explain_with_shap,
    generate_explanations,
)


def _synthetic_cluster_data() -> tuple[pd.DataFrame, list[int], list[str]]:
    """Build a synthetic dataset with a clear two-feature cluster split.

    Cluster 0: high feature_a, low feature_b.
    Cluster 1: low feature_a, high feature_b.
    This ensures SHAP/LIME explanations should rank feature_a and feature_b
    as the most important distinguishing features.
    """
    np.random.seed(42)
    n = 60
    feature_a = np.concatenate([np.random.normal(10, 0.5, n // 2),
                                np.random.normal(1, 0.5, n // 2)])
    feature_b = np.concatenate([np.random.normal(1, 0.5, n // 2),
                                np.random.normal(10, 0.5, n // 2)])
    # Add a noise feature that should not rank highly.
    feature_noise = np.random.normal(5, 2, n)

    labels = [0] * (n // 2) + [1] * (n // 2)
    feature_columns = ["feature_a", "feature_b", "feature_noise"]

    df = pd.DataFrame({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "feature_noise": feature_noise,
    })

    return df, labels, feature_columns


class SHAPExplainabilityTests(unittest.TestCase):
    """Tests for the SHAP surrogate-model explanation path."""

    def test_shap_explanations_rank_discriminative_features_first(self) -> None:
        """SHAP should rank the two discriminative features above noise."""
        df, labels, feature_columns = _synthetic_cluster_data()
        result = explain_with_shap(df, labels, feature_columns)

        self.assertIn("customer_contributions", result)
        self.assertIn("segment_aggregates", result)
        self.assertGreater(len(result["customer_contributions"]), 0)

        # Check that top contributions for each customer are feature_a/feature_b.
        for contrib_entry in result["customer_contributions"]:
            top_features = [c["feature"] for c in contrib_entry["contributions"][:2]]
            self.assertTrue(
                set(top_features) <= {"feature_a", "feature_b"},
                f"Expected discriminative features in top-2, got {top_features}",
            )

    def test_shap_segment_aggregates_present(self) -> None:
        """SHAP segment-level aggregates should be present for both clusters."""
        df, labels, feature_columns = _synthetic_cluster_data()
        result = explain_with_shap(df, labels, feature_columns)
        self.assertIn("0", result["segment_aggregates"])
        self.assertIn("1", result["segment_aggregates"])

        # Top segment-defining feature should be feature_a or feature_b.
        for seg_label in ("0", "1"):
            top_feature = result["segment_aggregates"][seg_label][0]["feature"]
            self.assertIn(top_feature, {"feature_a", "feature_b"})


class LIMEExplainabilityTests(unittest.TestCase):
    """Tests for the LIME surrogate-model explanation path."""

    def test_lime_explanations_rank_discriminative_features(self) -> None:
        """LIME should also rank discriminative features above noise."""
        df, labels, feature_columns = _synthetic_cluster_data()
        result = explain_with_lime(df, labels, feature_columns)

        self.assertIn("customer_contributions", result)
        self.assertGreater(len(result["customer_contributions"]), 0)

        # Check that the top features are discriminative.
        for contrib_entry in result["customer_contributions"][:5]:
            top_features = [c["feature"] for c in contrib_entry["contributions"][:2]]
            self.assertTrue(
                set(top_features) <= {"feature_a", "feature_b"},
                f"Expected discriminative features in top-2, got {top_features}",
            )

    def test_lime_output_shape_matches_shap(self) -> None:
        """LIME output must have the same top-level keys as SHAP output."""
        df, labels, feature_columns = _synthetic_cluster_data()
        shap_result = explain_with_shap(df, labels, feature_columns)
        lime_result = explain_with_lime(df, labels, feature_columns)

        self.assertEqual(
            set(shap_result.keys()),
            set(lime_result.keys()),
            "SHAP and LIME outputs must have identical top-level keys",
        )


class FallbackExplainabilityTests(unittest.TestCase):
    """Tests for the rule-based fallback when surrogate fitting fails."""

    def test_single_cluster_falls_back_to_rule_based(self) -> None:
        """When all points are in one cluster, surrogate fitting should fail
        and the system should fall back to rule-based explanations."""
        import tempfile

        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            "f1": np.random.normal(5, 1, n),
            "f2": np.random.normal(3, 1, n),
        })
        labels = [0] * n  # Single cluster — surrogate can't fit.

        # Write a temporary features CSV for the rule-based fallback.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            df.to_csv(tmp, index=False)
            features_path = tmp.name

        try:
            node_outputs = {
                "segmentation": {
                    "feature_columns": ["f1", "f2"],
                    "cluster_centers": [[5.0, 3.0]],
                    "labels": labels,
                    "customer_ids": [str(i) for i in range(n)],
                },
                "feature_engineering": {
                    "features_path": features_path,
                },
            }

            # Force SHAP mode to trigger the surrogate path.
            old_mode = os.environ.get("EXPLAINABILITY_MODE")
            os.environ["EXPLAINABILITY_MODE"] = "shap"
            try:
                result = generate_explanations(node_outputs)
            finally:
                if old_mode is None:
                    os.environ.pop("EXPLAINABILITY_MODE", None)
                else:
                    os.environ["EXPLAINABILITY_MODE"] = old_mode

            self.assertEqual(result["explainer_used"], "rule_based_fallback")
            self.assertIn("surrogate fit failure", result["explainer_reason"])
            # Rule-based fallback should still produce explanations.
            self.assertGreater(len(result["customer_explanations"]), 0)
        finally:
            os.unlink(features_path)


class ExplainabilityModeEnvVarTests(unittest.TestCase):
    """Tests for EXPLAINABILITY_MODE environment variable switching."""

    def test_lime_mode_via_env_var(self) -> None:
        """Setting EXPLAINABILITY_MODE=lime should invoke the LIME path."""
        import tempfile

        df, labels, feature_columns = _synthetic_cluster_data()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            df.to_csv(tmp, index=False)
            features_path = tmp.name

        try:
            # Build minimal cluster_centers for fallback (2 clusters).
            cluster_centers = [
                [float(df[col][labels.index(0)]) for col in feature_columns]
                for _ in range(2)
            ]

            node_outputs = {
                "segmentation": {
                    "feature_columns": feature_columns,
                    "cluster_centers": cluster_centers,
                    "labels": labels,
                    "customer_ids": [str(i) for i in range(len(labels))],
                },
                "feature_engineering": {
                    "features_path": features_path,
                },
            }

            old_mode = os.environ.get("EXPLAINABILITY_MODE")
            os.environ["EXPLAINABILITY_MODE"] = "lime"
            try:
                result = generate_explanations(node_outputs)
            finally:
                if old_mode is None:
                    os.environ.pop("EXPLAINABILITY_MODE", None)
                else:
                    os.environ["EXPLAINABILITY_MODE"] = old_mode

            self.assertEqual(result["explainer_used"], "lime")
            self.assertGreater(len(result["customer_explanations"]), 0)
        finally:
            os.unlink(features_path)

    def test_rule_based_mode_via_env_var(self) -> None:
        """Setting EXPLAINABILITY_MODE=rule_based uses the original logic."""
        import tempfile

        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            "f1": np.concatenate([np.random.normal(10, 1, n // 2),
                                  np.random.normal(1, 1, n // 2)]),
            "f2": np.concatenate([np.random.normal(1, 1, n // 2),
                                  np.random.normal(10, 1, n // 2)]),
        })
        labels = [0] * (n // 2) + [1] * (n // 2)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            df.to_csv(tmp, index=False)
            features_path = tmp.name

        try:
            node_outputs = {
                "segmentation": {
                    "feature_columns": ["f1", "f2"],
                    "cluster_centers": [[10.0, 1.0], [1.0, 10.0]],
                    "labels": labels,
                    "customer_ids": [str(i) for i in range(n)],
                },
                "feature_engineering": {
                    "features_path": features_path,
                },
            }

            old_mode = os.environ.get("EXPLAINABILITY_MODE")
            os.environ["EXPLAINABILITY_MODE"] = "rule_based"
            try:
                result = generate_explanations(node_outputs)
            finally:
                if old_mode is None:
                    os.environ.pop("EXPLAINABILITY_MODE", None)
                else:
                    os.environ["EXPLAINABILITY_MODE"] = old_mode

            self.assertEqual(result["explainer_used"], "rule_based")
            self.assertGreater(len(result["customer_explanations"]), 0)
        finally:
            os.unlink(features_path)


if __name__ == "__main__":
    unittest.main()
