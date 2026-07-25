"""ASTER visualization node module."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_visualization_payload(features: pd.DataFrame, labels: list[int]) -> dict[str, Any]:
    """Create a simple visualization payload for cluster exploration."""

    rows = []
    for index, row in features.iterrows():
        rows.append(
            {
                "x": round(float(row["credit_headroom"]), 4),
                "y": round(float(row["monthly_spend"]), 4),
                "label": int(labels[index]),
                "customer_id": str(row["CUST_ID"]),
            }
        )

    return {
        "chart_type": "scatter",
        "data": rows,
    }