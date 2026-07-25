"""ASTER response composer module."""

from __future__ import annotations

from typing import Any


def compose_response(
    workflow_name: str,
    intent: str,
    intent_classification: str,
    node_outputs: dict[str, Any],
    explanations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge collected node outputs into a single structured response object.

    Args:
        workflow_name: Name of the workflow that was executed.
        intent: The original intent from the query context.
        intent_classification: The classified intent (full_workflow, explanation_only, eda_only).
        node_outputs: Dictionary mapping node names to their execution outputs.

    Returns:
        A structured response containing statistics, recommendations, and visual outputs.
    """

    response: dict[str, Any] = {
        "workflow_name": workflow_name,
        "intent": intent,
        "intent_classification": intent_classification,
        "summary": {},
        "statistics": {},
        "recommendations": {},
        "visualizations": {},
        "explanations": explanations or {"customer_explanations": [], "segment_summaries": []},
        "metadata": {
            "nodes_executed": list(node_outputs.keys()),
            "node_count": len(node_outputs),
        },
    }

    # Merge analytics output
    if "analytics" in node_outputs:
        analytics_output = node_outputs["analytics"]
        response["statistics"]["descriptive"] = {
            "row_count": analytics_output.get("row_count"),
            "column_count": analytics_output.get("column_count"),
            "numeric_summary": analytics_output.get("numeric_summary"),
            "categorical_summary": analytics_output.get("categorical_summary"),
            "missing_values": analytics_output.get("missing_values"),
        }

    # Merge EDA output
    if "eda" in node_outputs:
        eda_output = node_outputs["eda"]
        response["statistics"]["exploratory"] = {
            "missing_values": eda_output.get("missing_values"),
            "missing_percentage": eda_output.get("missing_percentage"),
            "numeric_correlations": eda_output.get("numeric_correlations"),
            "sample_rows": eda_output.get("sample_rows"),
        }

    # Merge segmentation output
    if "segmentation" in node_outputs:
        seg_output = node_outputs["segmentation"]
        response["summary"]["segmentation"] = {
            "cluster_count": seg_output.get("cluster_count"),
            "inertia": seg_output.get("inertia"),
            "customer_count": len(seg_output.get("customer_ids", [])),
        }

    # Merge evaluation output
    if "evaluation" in node_outputs:
        eval_output = node_outputs["evaluation"]
        response["summary"]["evaluation"] = {
            "silhouette_score": eval_output.get("silhouette_score"),
            "cluster_sizes": eval_output.get("cluster_sizes"),
        }

    # Merge recommendation output
    if "recommendation" in node_outputs:
        rec_output = node_outputs["recommendation"]
        response["recommendations"] = {
            "cluster_recommendations": rec_output.get("cluster_recommendations"),
            "customer_recommendations": rec_output.get("customer_recommendations"),
            "cluster_tiers": rec_output.get("cluster_tiers"),
            "tier_counts": rec_output.get("tier_counts"),
        }

    # Merge visualization output
    if "visualization" in node_outputs:
        viz_output = node_outputs["visualization"]
        response["visualizations"] = {
            "scatter": viz_output.get("scatter"),
            "cluster_size_bar": viz_output.get("cluster_size_bar"),
        }

    # Merge feature engineering output
    if "feature_engineering" in node_outputs:
        fe_output = node_outputs["feature_engineering"]
        response["metadata"]["features_path"] = fe_output.get("features_path")
        response["metadata"]["feature_row_count"] = fe_output.get("row_count")

    return response
