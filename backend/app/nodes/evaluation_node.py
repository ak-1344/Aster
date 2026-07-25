"""ASTER evaluation node module."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import silhouette_score


def evaluate_segmentation(features: pd.DataFrame, labels: list[int]) -> dict[str, object]:
    """Evaluate a clustering result with a silhouette score and cluster size summary."""

    feature_columns = [column for column in features.columns if column != "CUST_ID"]
    matrix = features[feature_columns].fillna(0.0)
    score = silhouette_score(matrix, labels)
    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    return {
        "silhouette_score": round(float(score), 4),
        "cluster_sizes": {str(key): int(value) for key, value in cluster_sizes.items()},
    }