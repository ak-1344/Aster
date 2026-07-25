"""ASTER segmentation node module."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.cluster import KMeans


def segment_customers(features: pd.DataFrame, n_clusters: int = 2) -> dict[str, Any]:
    """Cluster customer features with a simple KMeans baseline."""

    feature_columns = [column for column in features.columns if column != "CUST_ID"]
    matrix = features[feature_columns].fillna(0.0)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    return {
        "labels": labels.tolist(),
        "cluster_count": int(n_clusters),
        "feature_columns": feature_columns,
        "customer_ids": features["CUST_ID"].astype(str).tolist(),
    }