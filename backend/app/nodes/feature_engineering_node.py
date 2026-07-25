"""Feature engineering node for ASTER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.llm.gemini_client import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiUnavailableError,
    request_structured_output,
)
from backend.utils.feature_engineering import build_customer_features


def generate_features(
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Generate customer-level features for downstream segmentation."""

    return build_customer_features(dataset_path=dataset_path, output_path=output_path)


def _feature_summary(features_df: pd.DataFrame) -> dict[str, Any]:
    """Build a lightweight summary of available engineered features."""

    feature_columns = [col for col in features_df.columns if col != "CUST_ID"]
    summary: dict[str, dict[str, float]] = {}
    for col in feature_columns:
        values = features_df[col]
        summary[col] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        }
    return {"feature_count": len(feature_columns), "features": summary}


def _selection_schema(feature_names: list[str]) -> dict[str, Any]:
    """Return the bounded JSON schema for feature relevance ranking."""

    return {
        "type": "object",
        "properties": {
            "selected_features": {
                "type": "array",
                "items": {"type": "string", "enum": feature_names},
                "description": "Ordered list of the most relevant features for this query.",
            },
            "reasoning": {
                "type": "string",
                "description": "A concise reason for the feature ranking.",
            },
        },
        "required": ["selected_features", "reasoning"],
        "additionalProperties": False,
    }


def _selection_prompt(
    feature_summary: dict[str, Any],
    query_context: dict[str, Any] | None,
) -> str:
    """Build a prompt that limits Gemini to ranking existing features."""

    query_summary = {
        "normalized_query": (query_context or {}).get("normalized_query", ""),
        "intent": (query_context or {}).get("intent", "segmentation"),
        "filters": (query_context or {}).get("filters", {}),
    }
    return "\n".join(
        [
            "You assist ASTER's feature engineering node.",
            "Rank the already-computed features by relevance to the user's query intent.",
            "You must not compute new features, transform data, or invent feature names.",
            "Return only features from the supplied list, ordered by relevance.",
            f"Query context: {json.dumps(query_summary, sort_keys=True)}",
            f"Available features: {json.dumps(feature_summary, sort_keys=True)}",
        ]
    )


def _default_feature_selection(features_df: pd.DataFrame) -> list[str]:
    """Return all non-ID features in their original column order."""

    return [col for col in features_df.columns if col != "CUST_ID"]


def select_features_to_surface(
    features_df: pd.DataFrame,
    query_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use Gemini to rank which already-engineered features are most relevant.

    Gemini may only select and rank features from the existing DataFrame.
    It does not compute new features or transform data.
    Falls back to returning all features in their original order.
    """

    feature_columns = [col for col in features_df.columns if col != "CUST_ID"]
    feature_summary = _feature_summary(features_df)

    try:
        payload = request_structured_output(
            _selection_prompt(feature_summary, query_context),
            _selection_schema(sorted(feature_columns)),
        )
        selected = payload.get("selected_features")
        reasoning = payload.get("reasoning")

        if not isinstance(selected, list) or not selected:
            raise ValueError("selected_features must be a non-empty list")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("reasoning must be a non-empty string")

        # Filter to only valid feature names
        validated = [f for f in selected if f in feature_columns]
        if not validated:
            raise ValueError("No valid features in Gemini selection")

        return {
            "selected_features": validated,
            "all_features": feature_columns,
            "reasoning": reasoning.strip(),
            "path": "llm_assisted",
        }
    except (
        GeminiUnavailableError,
        GeminiRequestError,
        GeminiResponseError,
        ValueError,
    ) as error:
        return {
            "selected_features": _default_feature_selection(features_df),
            "all_features": feature_columns,
            "reasoning": (
                f"All features returned in default order because Gemini feature "
                f"ranking was unavailable: {error}"
            ),
            "path": "rule_based_fallback",
        }