"""ASTER recommendation node module."""

from __future__ import annotations

import pandas as pd


def build_recommendations(features: pd.DataFrame) -> list[dict[str, object]]:
    """Create simple, rule-based recommendations from engineered features."""

    ranked = features.sort_values(by=["monthly_spend", "credit_headroom"], ascending=False)
    recommendations = []
    for _, row in ranked.head(5).iterrows():
        recommendations.append(
            {
                "customer_id": str(row["CUST_ID"]),
                "monthly_spend": round(float(row["monthly_spend"]), 4),
                "credit_headroom": round(float(row["credit_headroom"]), 4),
                "action": "prioritize_retention",
            }
        )

    return recommendations