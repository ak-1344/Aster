"""ASTER segmentation node module."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.app.model_registry import get


def segment_customers(features: pd.DataFrame, n_clusters: int = 3) -> dict[str, Any]:
    """Cluster customer features with a simple KMeans baseline."""

    feature_columns = [column for column in features.columns if column != "CUST_ID"]
    matrix = features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    kmeans_factory = get("kmeans")
    if kmeans_factory is None:
        raise RuntimeError("KMeans factory not found in model registry")
    model = kmeans_factory(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    clustered = features.copy().reset_index(drop=True)
    clustered["cluster_label"] = labels

    return {
        "labels": labels.tolist(),
        "cluster_count": int(n_clusters),
        "feature_columns": feature_columns,
        "customer_ids": features["CUST_ID"].astype(str).tolist(),
        "clustered_customers": clustered[["CUST_ID", "cluster_label"]].to_dict(orient="records"),
        "cluster_centers": model.cluster_centers_.round(4).tolist(),
        "inertia": round(float(model.inertia_), 4),
    }