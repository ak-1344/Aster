"""ASTER evaluation node module."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import silhouette_score


def evaluate_segmentation(features: pd.DataFrame, labels: list[int]) -> dict[str, object]:
    """Evaluate a clustering result with a silhouette score and cluster size summary."""

    feature_columns = [column for column in features.columns if column != "CUST_ID"]
    matrix = features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    score = None
    if len(set(labels)) > 1 and len(matrix) > len(set(labels)):
        score = silhouette_score(matrix, labels)

    return {
        "silhouette_score": None if score is None else round(float(score), 4),
        "cluster_sizes": {str(key): int(value) for key, value in cluster_sizes.items()},
        "cluster_count": int(len(set(labels))),
        "customer_count": int(len(labels)),
    }