"""ASTER visualization node module."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_visualization_payload(features: pd.DataFrame, labels: list[int]) -> dict[str, Any]:
    """Create Plotly-ready cluster scatter and size bar payloads."""

    clustered = features.copy().reset_index(drop=True)
    clustered["cluster_label"] = labels

    scatter_rows = []
    for _, row in clustered.iterrows():
        scatter_rows.append(
            {
                "x": round(float(row["credit_headroom"]), 4),
                "y": round(float(row["monthly_spend"]), 4),
                "cluster_label": int(row["cluster_label"]),
                "customer_id": str(row["CUST_ID"]),
            }
        )

    cluster_sizes = clustered["cluster_label"].value_counts().sort_index()

    return {
        "scatter": {
            "chart_type": "scatter",
            "title": "Customer Clusters",
            "x_axis": "credit_headroom",
            "y_axis": "monthly_spend",
            "data": scatter_rows,
        },
        "cluster_size_bar": {
            "chart_type": "bar",
            "title": "Cluster Sizes",
            "x_axis": "cluster_label",
            "y_axis": "customer_count",
            "data": [
                {"cluster_label": int(cluster_label), "customer_count": int(count)}
                for cluster_label, count in cluster_sizes.items()
            ],
        },
    }