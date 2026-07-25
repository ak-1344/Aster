"""ASTER scheduler module."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.app.execution_graph import ExecutionGraph
from backend.app.nodes import (
    analytics_node,
    eda_node,
    evaluation_node,
    feature_engineering_node,
    recommendation_node,
    segmentation_node,
    visualization_node,
)


class SchedulerExecutionError(Exception):
    """Raised when a node execution fails."""


def _execute_node(node_name: str, context: dict[str, Any]) -> dict[str, Any]:
    """Execute a single node and return its output."""

    try:
        if node_name == "analytics":
            return analytics_node.build_descriptive_statistics(
                dataset_path=context.get("dataset_path")
            )
        elif node_name == "eda":
            return eda_node.build_exploratory_summary(
                dataset_path=context.get("dataset_path")
            )
        elif node_name == "feature_engineering":
            features_df, output_path = feature_engineering_node.generate_features(
                dataset_path=context.get("dataset_path"),
                output_path=context.get("output_path"),
            )
            context["customer_features"] = features_df
            context["features_path"] = str(output_path)
            return {"features_path": str(output_path), "row_count": len(features_df)}
        elif node_name == "segmentation":
            features = context.get("customer_features")
            if features is None:
                raise SchedulerExecutionError("customer_features not found in context for segmentation")
            n_clusters = context.get("n_clusters", 3)
            result = segmentation_node.segment_customers(features, n_clusters=n_clusters)
            context["cluster_labels"] = result["labels"]
            context["clustered_customers"] = result["clustered_customers"]
            return result
        elif node_name == "evaluation":
            features = context.get("customer_features")
            labels = context.get("cluster_labels")
            if features is None or labels is None:
                raise SchedulerExecutionError("customer_features or cluster_labels not found in context for evaluation")
            return evaluation_node.evaluate_segmentation(features, labels)
        elif node_name == "recommendation":
            features = context.get("customer_features")
            labels = context.get("cluster_labels")
            if features is None or labels is None:
                raise SchedulerExecutionError("customer_features or cluster_labels not found in context for recommendation")
            return recommendation_node.build_recommendations(features, labels)
        elif node_name == "visualization":
            features = context.get("customer_features")
            labels = context.get("cluster_labels")
            if features is None or labels is None:
                raise SchedulerExecutionError("customer_features or cluster_labels not found in context for visualization")
            return visualization_node.build_visualization_payload(features, labels)
        else:
            raise SchedulerExecutionError(f"Unknown node: {node_name}")
    except Exception as e:
        raise SchedulerExecutionError(f"Failed to execute node {node_name}: {e}") from e


def execute_graph(graph: ExecutionGraph, initial_context: dict[str, Any]) -> dict[str, Any]:
    """Execute the execution graph in dependency order and collect node outputs.

    Args:
        graph: The ExecutionGraph to execute.
        initial_context: Initial context containing dataset_path, n_clusters, etc.

    Returns:
        A dictionary mapping node names to their execution outputs.

    Raises:
        SchedulerExecutionError: If any node execution fails.
    """

    execution_order = graph.get_execution_order()
    context = initial_context
    outputs: dict[str, Any] = {}

    for node_name in execution_order:
        node_output = _execute_node(node_name, context)
        outputs[node_name] = node_output

    return outputs
