"""Deterministic customer segmentation with bounded Gemini configuration assistance."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from backend.app.llm.gemini_client import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiUnavailableError,
    request_structured_output,
)
from backend.app.model_registry import get, list_available


def _data_profile(matrix: pd.DataFrame) -> dict[str, Any]:
    """Create a lightweight numeric profile suitable for algorithm selection."""

    feature_ranges: dict[str, dict[str, float]] = {}
    for column in matrix.columns:
        values = matrix[column]
        feature_ranges[column] = {
            "minimum": round(float(values.min()), 4),
            "maximum": round(float(values.max()), 4),
            "variance": round(float(values.var()), 4),
        }

    return {
        "row_count": int(len(matrix)),
        "feature_count": int(len(matrix.columns)),
        "feature_ranges": feature_ranges,
    }


def _active_clustering_algorithms() -> dict[str, dict[str, Any]]:
    """Return only registry algorithms that can execute in this environment."""

    return {
        entry["name"]: entry["metadata"]
        for entry in list_available()
        if entry["metadata"].get("type") == "clustering"
        and entry["metadata"].get("status") == "active"
        and get(entry["name"]) is not None
    }


def _selection_schema(algorithms: list[str]) -> dict[str, Any]:
    """Return the bounded JSON schema for a clustering configuration suggestion."""

    return {
        "type": "object",
        "properties": {
            "algorithm": {"type": "string", "enum": algorithms},
            "parameters": {
                "type": "object",
                "properties": {
                    "n_clusters": {"type": "integer"},
                    "eps": {"type": "number"},
                    "min_samples": {"type": "integer"},
                    "min_cluster_size": {"type": "integer"},
                },
            },
            "reasoning": {
                "type": "string",
                "description": "A concise reason based only on the supplied data profile.",
            },
        },
        "required": ["algorithm", "parameters", "reasoning"],
        "additionalProperties": False,
    }


def _selection_prompt(
    profile: dict[str, Any],
    available_algorithms: dict[str, dict[str, Any]],
    query_context: dict[str, Any] | None,
) -> str:
    """Build a prompt that limits Gemini to algorithm and parameter selection."""

    query_summary = {
        "normalized_query": (query_context or {}).get("normalized_query", ""),
        "intent": (query_context or {}).get("intent", "segmentation"),
        "filters": (query_context or {}).get("filters", {}),
    }
    return "\n".join(
        [
            "You assist ASTER's deterministic segmentation node.",
            "Select one active algorithm and conservative parameter values from the supplied registry.",
            "You must not assign labels, describe individual customers, calculate scores, or invent algorithms.",
            "The selected algorithm will be executed deterministically by the model registry.",
            f"Query context: {json.dumps(query_summary, sort_keys=True)}",
            f"Active registry algorithms: {json.dumps(available_algorithms, sort_keys=True)}",
            f"Lightweight data profile: {json.dumps(profile, sort_keys=True)}",
        ]
    )


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    """Coerce an LLM-suggested integer into a safe inclusive range."""

    try:
        candidate = int(value)
    except (TypeError, ValueError):
        candidate = default
    return max(minimum, min(candidate, maximum))


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    """Coerce an LLM-suggested float into a safe inclusive range."""

    try:
        candidate = float(value)
    except (TypeError, ValueError):
        candidate = default
    return max(minimum, min(candidate, maximum))


def _safe_kmeans_cluster_count(row_count: int, requested_clusters: int) -> int:
    """Keep KMeans cluster counts executable for the available row count."""

    if row_count < 2:
        raise ValueError("At least two rows are required for segmentation")
    return _as_int(requested_clusters, 3, 2, row_count - 1)


def _sanitize_parameters(
    algorithm: str,
    parameters: dict[str, Any],
    row_count: int,
    requested_clusters: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Allow only bounded, algorithm-specific parameters from Gemini."""

    if algorithm == "kmeans":
        return {
            "n_clusters": _safe_kmeans_cluster_count(
                row_count,
                parameters.get("n_clusters", requested_clusters),
            ),
            "random_state": 42,
            "n_init": 10,
        }

    if algorithm == "dbscan":
        maximum_range = max(
            (details["maximum"] - details["minimum"])
            for details in profile["feature_ranges"].values()
        )
        return {
            "eps": _as_float(parameters.get("eps"), 0.5, 0.0001, max(maximum_range, 0.0001)),
            "min_samples": _as_int(
                parameters.get("min_samples"),
                5,
                2,
                row_count,
            ),
        }

    if algorithm == "hdbscan":
        return {
            "min_cluster_size": _as_int(
                parameters.get("min_cluster_size"),
                5,
                2,
                row_count,
            ),
            "min_samples": _as_int(
                parameters.get("min_samples"),
                5,
                2,
                row_count,
            ),
        }

    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")


def _kmeans_fallback(
    row_count: int,
    requested_clusters: int,
    reason: str,
    *,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Return the deterministic baseline selection used when Gemini is unavailable."""

    return {
        "algorithm": "kmeans",
        "parameters": {
            "n_clusters": _safe_kmeans_cluster_count(row_count, requested_clusters),
            "random_state": 42,
            "n_init": 10,
        },
        "reason": reason,
        "path": "rule_based_fallback",
        "data_profile": profile,
    }


def select_segmentation_strategy(
    matrix: pd.DataFrame,
    n_clusters: int,
    query_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask Gemini for bounded configuration advice, with a KMeans fallback."""

    profile = _data_profile(matrix)
    available_algorithms = _active_clustering_algorithms()
    if "kmeans" not in available_algorithms:
        raise RuntimeError("KMeans must be active in the model registry")

    try:
        payload = request_structured_output(
            _selection_prompt(profile, available_algorithms, query_context),
            _selection_schema(sorted(available_algorithms)),
        )
        algorithm = payload.get("algorithm")
        parameters = payload.get("parameters")
        reasoning = payload.get("reasoning")
        if algorithm not in available_algorithms:
            raise ValueError("Gemini selected an unavailable algorithm")
        if not isinstance(parameters, dict):
            raise ValueError("Gemini parameters must be an object")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("Gemini reasoning must be a non-empty string")

        return {
            "algorithm": algorithm,
            "parameters": _sanitize_parameters(
                algorithm,
                parameters,
                len(matrix),
                n_clusters,
                profile,
            ),
            "reason": reasoning.strip(),
            "path": "llm_assisted",
            "data_profile": profile,
        }
    except (GeminiUnavailableError, GeminiRequestError, GeminiResponseError, ValueError) as error:
        return _kmeans_fallback(
            len(matrix),
            n_clusters,
            f"Rule-based fallback selected KMeans because Gemini selection was unavailable or invalid: {error}",
            profile=profile,
        )


def _fit_selected_model(
    algorithm: str,
    parameters: dict[str, Any],
    matrix: pd.DataFrame,
) -> tuple[Any, list[int]]:
    """Fit a registry model and convert its deterministic labels to Python integers."""

    factory = get(algorithm)
    if factory is None:
        raise RuntimeError(f"{algorithm} factory not found in model registry")

    model = factory(**parameters)
    labels = [int(label) for label in model.fit_predict(matrix)]
    return model, labels


def _non_noise_cluster_count(labels: list[int]) -> int:
    """Count real clusters while excluding the DBSCAN/HDBSCAN noise label."""

    return len({label for label in labels if label != -1})


def _explanation_centers(matrix: pd.DataFrame, labels: list[int]) -> dict[str, list[float]]:
    """Build deterministic per-label centers for explainability across algorithms."""

    centered = matrix.copy()
    centered["cluster_label"] = labels
    return {
        str(int(label)): [round(float(value), 4) for value in center]
        for label, center in centered.groupby("cluster_label")[matrix.columns].mean().iterrows()
    }


def _compute_customer_boundary_metrics(
    matrix: pd.DataFrame,
    labels: list[int],
    customer_ids: list[str],
    explanation_centers: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Compute per-customer distance to the second-nearest cluster centroid."""

    import numpy as np

    center_labels = sorted(int(label) for label in explanation_centers if int(label) >= 0)
    if len(center_labels) < 2:
        return []

    center_matrix = np.array(
        [explanation_centers[str(label)] for label in center_labels],
        dtype=float,
    )
    label_to_index = {label: index for index, label in enumerate(center_labels)}
    rows = matrix.to_numpy(dtype=float)
    metrics: list[dict[str, Any]] = []

    for row, cluster_label, customer_id in zip(rows, labels, customer_ids):
        if cluster_label < 0 or cluster_label not in label_to_index:
            continue

        distances = np.linalg.norm(center_matrix - row, axis=1)
        own_index = label_to_index[cluster_label]
        own_distance = float(distances[own_index])

        alternate_distances = [
            (float(distances[index]), center_labels[index])
            for index in range(len(center_labels))
            if index != own_index
        ]
        if not alternate_distances:
            continue

        second_distance, alternate_label = min(alternate_distances, key=lambda item: item[0])
        ratio = own_distance / second_distance if second_distance > 0 else 1.0

        metrics.append(
            {
                "customer_id": customer_id,
                "cluster_label": int(cluster_label),
                "nearest_alternate_cluster": int(alternate_label),
                "boundary_distance_ratio": round(ratio, 4),
            }
        )

    return metrics


def segment_customers(
    features: pd.DataFrame,
    n_clusters: int = 3,
    query_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cluster features deterministically using a Gemini-assisted safe configuration."""

    feature_columns = [column for column in features.columns if column != "CUST_ID"]
    matrix = features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    selection = select_segmentation_strategy(matrix, n_clusters, query_context)
    algorithm = selection["algorithm"]
    parameters = selection["parameters"]

    try:
        model, labels = _fit_selected_model(algorithm, parameters, matrix)
    except Exception as error:
        selection = _kmeans_fallback(
            len(matrix),
            n_clusters,
            f"Rule-based fallback selected KMeans after {algorithm} could not execute: {type(error).__name__}",
            profile=selection["data_profile"],
        )
        algorithm = selection["algorithm"]
        parameters = selection["parameters"]
        model, labels = _fit_selected_model(algorithm, parameters, matrix)

    if algorithm != "kmeans" and _non_noise_cluster_count(labels) < 2:
        previous_algorithm = algorithm
        selection = _kmeans_fallback(
            len(matrix),
            n_clusters,
            (
                f"Rule-based fallback selected KMeans because {previous_algorithm} produced "
                "fewer than two usable clusters."
            ),
            profile=selection["data_profile"],
        )
        algorithm = selection["algorithm"]
        parameters = selection["parameters"]
        model, labels = _fit_selected_model(algorithm, parameters, matrix)

    clustered = features.copy().reset_index(drop=True)
    clustered["cluster_label"] = labels
    cluster_centers = getattr(model, "cluster_centers_", None)
    inertia = getattr(model, "inertia_", None)
    explanation_centers = _explanation_centers(matrix, labels)
    customer_boundary_metrics = _compute_customer_boundary_metrics(
        matrix,
        labels,
        features["CUST_ID"].astype(str).tolist(),
        explanation_centers,
    )

    return {
        "labels": labels,
        "cluster_count": _non_noise_cluster_count(labels),
        "feature_columns": feature_columns,
        "customer_ids": features["CUST_ID"].astype(str).tolist(),
        "clustered_customers": clustered[["CUST_ID", "cluster_label"]].to_dict(orient="records"),
        "cluster_centers": (
            cluster_centers.round(4).tolist() if cluster_centers is not None else None
        ),
        "explanation_centers": explanation_centers,
        "customer_boundary_metrics": customer_boundary_metrics,
        "inertia": None if inertia is None else round(float(inertia), 4),
        "chosen_algorithm": algorithm,
        "algorithm_parameters": parameters,
        "algorithm_reason": selection["reason"],
        "llm_assistance": selection,
    }
