"""ASTER recommendation node module."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd

from backend.app.llm.gemini_client import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiUnavailableError,
    request_structured_output,
)
from backend.app.model_registry import register


def _compute_tier_thresholds(features: pd.DataFrame) -> dict[str, float]:
    """Compute dataset-wide quantile thresholds for tier assignment."""

    return {
        "spend_q67": float(features["monthly_spend"].quantile(0.67)),
        "spend_q33": float(features["monthly_spend"].quantile(0.33)),
        "frequency_q67": float(features["transaction_frequency_per_month"].quantile(0.67)),
        "frequency_q33": float(features["transaction_frequency_per_month"].quantile(0.33)),
        "credit_headroom_median": float(features["credit_headroom"].median()),
    }


def _tier_from_cluster_profile(
    cluster_frame: pd.DataFrame,
    thresholds: dict[str, float],
) -> str:
    """Map a cluster profile to a business tier using dataset-wide thresholds."""

    spend = float(cluster_frame["monthly_spend"].median())
    frequency = float(cluster_frame["transaction_frequency_per_month"].median())
    payment_ratio = float(cluster_frame["full_payment_ratio"].median())
    headroom = float(cluster_frame["credit_headroom"].median())

    if spend >= thresholds["spend_q67"] and frequency >= thresholds["frequency_q67"]:
        return "priority"
    if spend <= thresholds["spend_q33"] or frequency <= thresholds["frequency_q33"]:
        return "dormant"
    if payment_ratio >= 0.75 and headroom >= thresholds["credit_headroom_median"]:
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


def _behavioral_product_tags(cluster_frame: pd.DataFrame, tier: str) -> list[str]:
    """Map cluster behavioral medians to specific bank product tags."""

    tags: list[str] = []
    util_median = float(cluster_frame["credit_utilization_ratio"].median())
    full_payment_median = float(cluster_frame["full_payment_ratio"].median())
    cash_advance_median = float(cluster_frame["cash_advance_ratio"].median())
    oneoff_median = float(cluster_frame["oneoff_purchase_share"].median())
    freq_median = float(cluster_frame["transaction_frequency_per_month"].median())
    freq_q33 = float(cluster_frame["transaction_frequency_per_month"].quantile(0.33))

    if util_median >= 0.6 and full_payment_median <= 0.5:
        tags.append("debt_consolidation_loan")

    if cash_advance_median >= 0.25:
        tags.append("cash_advance_alternative_credit_line")

    if oneoff_median >= 0.35 and full_payment_median >= 0.9:
        tags.append("premium_rewards_card")

    # Dormant tier already carries a reactivation-focused bundle; skip duplicate tag.
    if tier != "dormant" and freq_median <= freq_q33:
        tags.append("reactivation_offer")

    return tags


def _merge_cross_sell(tier_bundle: list[str], behavioral_tags: list[str]) -> list[str]:
    """Combine tier defaults with behavioral product tags without duplication."""

    merged: list[str] = []
    for item in tier_bundle + behavioral_tags:
        if item not in merged:
            merged.append(item)
    return merged


def _phrasing_schema() -> dict[str, Any]:
    """Return the bounded JSON schema for recommendation phrasing."""

    return {
        "type": "object",
        "properties": {
            "cluster_narratives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_label": {"type": "integer"},
                        "narrative": {
                            "type": "string",
                            "description": "A concise, human-readable business narrative for this cluster.",
                        },
                    },
                    "required": ["cluster_label", "narrative"],
                },
            },
        },
        "required": ["cluster_narratives"],
        "additionalProperties": False,
    }


def _phrasing_prompt(
    cluster_recommendations: list[dict[str, object]],
    query_context: dict[str, Any] | None,
) -> str:
    """Build a prompt that limits Gemini to phrasing the rule-engine output."""

    query_summary = {
        "normalized_query": (query_context or {}).get("normalized_query", ""),
        "intent": (query_context or {}).get("intent", "segmentation"),
        "unsupported_filters": (query_context or {}).get("unsupported_filters", []),
    }
    return "\n".join(
        [
            "You assist ASTER's recommendation node.",
            "The rule engine has already decided which recommendation applies to which cluster.",
            "Your only job is to rephrase the provided recommendation output into concise,",
            "business-friendly narratives. Do not change the tier, action, or cross-sell items.",
            "Do not invent new recommendations, change cluster assignments, or override the rule engine.",
            "IMPORTANT: If the user requested filters that are unsupported (see unsupported_filters in query context), DO NOT mention them in your narratives. The results reflect only the supported criteria.",
            f"Query context: {json.dumps(query_summary, sort_keys=True)}",
            f"Rule-engine recommendations: {json.dumps(cluster_recommendations, sort_keys=True, default=str)}",
        ]
    )


def _generate_narratives(
    cluster_recommendations: list[dict[str, object]],
    query_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Ask Gemini for human-readable phrasing of rule-engine recommendations.

    Gemini may only generate the human-readable phrasing/reasoning wrapped
    around the rule output — it never picks the recommendation.
    """

    try:
        payload = request_structured_output(
            _phrasing_prompt(cluster_recommendations, query_context),
            _phrasing_schema(),
        )
        narratives_raw = payload.get("cluster_narratives", [])
        if not isinstance(narratives_raw, list):
            raise ValueError("cluster_narratives must be a list")

        narratives: dict[int, str] = {}
        for entry in narratives_raw:
            label = entry.get("cluster_label")
            narrative = entry.get("narrative")
            if isinstance(label, int) and isinstance(narrative, str) and narrative.strip():
                narratives[label] = narrative.strip()

        if not narratives:
            raise ValueError("No valid narratives returned by Gemini")

        return {
            "narratives": narratives,
            "path": "llm_assisted",
            "reasoning": "Gemini rephrased rule-engine recommendations into business narratives.",
        }
    except (
        GeminiUnavailableError,
        GeminiRequestError,
        GeminiResponseError,
        ValueError,
    ) as error:
        return {
            "narratives": {},
            "path": "rule_based_fallback",
            "reasoning": (
                f"Rule-engine messages used as-is because Gemini phrasing was "
                f"unavailable: {error}"
            ),
        }


def build_recommendations(
    features: pd.DataFrame,
    labels: list[int],
    query_context: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Create rule-based recommendations keyed off cluster labels."""

    clustered = features.copy().reset_index(drop=True)
    clustered["cluster_label"] = labels
    tier_thresholds = _compute_tier_thresholds(features)

    cluster_profiles: dict[int, dict[str, object]] = {}
    cluster_tiers: dict[int, str] = {}
    cluster_recommendations: list[dict[str, object]] = []

    for cluster_label, cluster_frame in clustered.groupby("cluster_label"):
        tier = _tier_from_cluster_profile(cluster_frame, tier_thresholds)
        cluster_tiers[int(cluster_label)] = tier
        bundle = _cluster_action_bundle(tier)
        behavioral_tags = _behavioral_product_tags(cluster_frame, tier)
        cross_sell = _merge_cross_sell(list(bundle["cross_sell"]), behavioral_tags)
        cluster_profiles[int(cluster_label)] = {
            "customer_count": int(len(cluster_frame)),
            "monthly_spend_median": round(float(cluster_frame["monthly_spend"].median()), 4),
            "transaction_frequency_median": round(float(cluster_frame["transaction_frequency_per_month"].median()), 4),
            "full_payment_ratio_median": round(float(cluster_frame["full_payment_ratio"].median()), 4),
            "tier": tier,
            "cross_sell": cross_sell,
            "behavioral_products": behavioral_tags,
        }
        cluster_recommendations.append(
            {
                "cluster_label": int(cluster_label),
                "tier": tier,
                "action": bundle["action"],
                "message": bundle["message"],
                "cross_sell": cross_sell,
                "behavioral_products": behavioral_tags,
                "customer_count": int(len(cluster_frame)),
            }
        )

    # Ask Gemini for human-readable phrasing of the rule-engine output
    narrative_result = _generate_narratives(cluster_recommendations, query_context)
    narratives = narrative_result.get("narratives", {})

    # Attach narratives to cluster recommendations
    for rec in cluster_recommendations:
        label = rec["cluster_label"]
        rec["narrative"] = narratives.get(label, rec["message"])

    customer_recommendations = []
    for _, row in clustered.iterrows():
        tier = cluster_tiers[int(row["cluster_label"])]
        cluster_label = int(row["cluster_label"])
        rec = next(r for r in cluster_recommendations if r["cluster_label"] == cluster_label)
        customer_recommendations.append(
            {
                "customer_id": str(row["CUST_ID"]),
                "cluster_label": cluster_label,
                "tier": tier,
                "primary_recommendation": rec["message"],
                "behavioral_products": rec.get("behavioral_products", []),
            }
        )

    return {
        "cluster_recommendations": cluster_recommendations,
        "customer_recommendations": customer_recommendations,
        "cluster_tiers": cluster_tiers,
        "tier_counts": {tier: int(count) for tier, count in Counter(cluster_tiers.values()).items()},
        "cluster_profiles": cluster_profiles,
        "llm_assistance": narrative_result,
    }


def _register_rule_engine() -> None:
    """Register the rule-based recommendation engine in the model registry."""
    register(
        "rule_engine",
        build_recommendations,
        {
            "type": "recommendation",
            "algorithm": "RuleEngine",
            "description": "Rule-based recommendation engine for customer segments",
            "status": "active",
        },
    )


_register_rule_engine()