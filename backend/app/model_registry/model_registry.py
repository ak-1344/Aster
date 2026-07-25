"""ASTER model registry module."""

from __future__ import annotations

from typing import Any, Callable

registry: dict[str, dict[str, Any]] = {}


def register(name: str, factory_fn: Callable[..., Any], metadata: dict[str, Any]) -> None:
    """Register a model or algorithm with its factory function and metadata."""
    registry[name] = {
        "factory_fn": factory_fn,
        "metadata": metadata,
    }


def get(name: str) -> Callable[..., Any] | None:
    """Retrieve a model factory function by name."""
    entry = registry.get(name)
    if entry is None:
        return None
    return entry["factory_fn"]


def list_available() -> list[dict[str, Any]]:
    """List all registered models with their metadata."""
    return [
        {"name": name, "metadata": entry["metadata"]} for name, entry in registry.items()
    ]


def _initialize_registry() -> None:
    """Initialize the model registry with baseline algorithms."""
    from sklearn.cluster import DBSCAN, KMeans

    def kmeans_factory(n_clusters: int = 3, random_state: int = 42, n_init: int = 10) -> KMeans:
        """Factory function for KMeans clustering."""
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    register(
        "kmeans",
        kmeans_factory,
        {
            "type": "clustering",
            "algorithm": "KMeans",
            "description": "K-Means clustering algorithm",
            "status": "active",
        },
    )

    register(
        "dbscan",
        lambda: None,
        {
            "type": "clustering",
            "algorithm": "DBSCAN",
            "description": "Density-based spatial clustering",
            "status": "registered_only",
        },
    )

    register(
        "hdbscan",
        lambda: None,
        {
            "type": "clustering",
            "algorithm": "HDBSCAN",
            "description": "Hierarchical density-based clustering",
            "status": "registered_only",
        },
    )

    register(
        "rule_engine",
        lambda: None,
        {
            "type": "recommendation",
            "algorithm": "RuleEngine",
            "description": "Rule-based recommendation engine",
            "status": "registered_only",
        },
    )


_initialize_registry()
