"""ASTER recommendation node module."""

from __future__ import annotations

from collections import Counter

import pandas as pd


def _tier_from_cluster_profile(cluster_frame: pd.DataFrame) -> str:
    """Map a cluster profile to a business tier."""

    spend = float(cluster_frame["monthly_spend"].median())
    frequency = float(cluster_frame["transaction_frequency_per_month"].median())
    payment_ratio = float(cluster_frame["full_payment_ratio"].median())
    headroom = float(cluster_frame["credit_headroom"].median())

    if spend >= cluster_frame["monthly_spend"].quantile(0.67) and frequency >= cluster_frame["transaction_frequency_per_month"].quantile(0.67):
        return "priority"
    if spend <= cluster_frame["monthly_spend"].quantile(0.33) or frequency <= cluster_frame["transaction_frequency_per_month"].quantile(0.33):
        return "dormant"
    if payment_ratio >= 0.75 and headroom >= cluster_frame["credit_headroom"].median():
        return "regular"
    return "regular"


def _cluster_action_bundle(tier: str) -> dict[str, object]:
    """Return a simple cross-sell and retention bundle for a tier."""

    bundles = {
        "priority": {
            "action": "retain_and_expand",
            "message": "Prioritize concierge retention and premium cross-sell.",
            "cross_sell": ["premium_card_upgrade", "travel_benefits", "cashback_bundle"],
        },
        "regular": {
            "action": "nurture_and_cross_sell",
            "message": "Encourage recurring use and targeted cross-sell.",
            "cross_sell": ["installment_offer", "auto_pay_enrollment", "category_rewards"],
        },
        "dormant": {
            "action": "reactivation",
            "message": "Use win-back nudges and low-friction offers.",
            "cross_sell": ["fee_waiver_offer", "light_spend_campaign", "welcome_back_promo"],
        },
    }
    return bundles[tier]


def build_recommendations(features: pd.DataFrame, labels: list[int]) -> dict[str, object]:
    """Create rule-based recommendations keyed off cluster labels."""

    clustered = features.copy().reset_index(drop=True)
    clustered["cluster_label"] = labels

    cluster_profiles: dict[int, dict[str, object]] = {}
    cluster_tiers: dict[int, str] = {}
    cluster_recommendations: list[dict[str, object]] = []

    for cluster_label, cluster_frame in clustered.groupby("cluster_label"):
        tier = _tier_from_cluster_profile(cluster_frame)
        cluster_tiers[int(cluster_label)] = tier
        bundle = _cluster_action_bundle(tier)
        cluster_profiles[int(cluster_label)] = {
            "customer_count": int(len(cluster_frame)),
            "monthly_spend_median": round(float(cluster_frame["monthly_spend"].median()), 4),
            "transaction_frequency_median": round(float(cluster_frame["transaction_frequency_per_month"].median()), 4),
            "full_payment_ratio_median": round(float(cluster_frame["full_payment_ratio"].median()), 4),
            "tier": tier,
            "cross_sell": bundle["cross_sell"],
        }
        cluster_recommendations.append(
            {
                "cluster_label": int(cluster_label),
                "tier": tier,
                "action": bundle["action"],
                "message": bundle["message"],
                "cross_sell": bundle["cross_sell"],
                "customer_count": int(len(cluster_frame)),
            }
        )

    customer_recommendations = []
    for _, row in clustered.iterrows():
        tier = cluster_tiers[int(row["cluster_label"])]
        customer_recommendations.append(
            {
                "customer_id": str(row["CUST_ID"]),
                "cluster_label": int(row["cluster_label"]),
                "tier": tier,
                "primary_recommendation": _cluster_action_bundle(tier)["message"],
            }
        )

    return {
        "cluster_recommendations": cluster_recommendations,
        "customer_recommendations": customer_recommendations,
        "cluster_tiers": cluster_tiers,
        "tier_counts": {tier: int(count) for tier, count in Counter(cluster_tiers.values()).items()},
        "cluster_profiles": cluster_profiles,
    }