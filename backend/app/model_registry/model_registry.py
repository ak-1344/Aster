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

    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        HDBSCAN = None

    def kmeans_factory(n_clusters: int = 3, random_state: int = 42, n_init: int = 10) -> KMeans:
        """Factory function for KMeans clustering."""
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    def dbscan_factory(eps: float = 0.5, min_samples: int = 5) -> DBSCAN:
        """Factory function for deterministic DBSCAN clustering."""
        return DBSCAN(eps=eps, min_samples=min_samples)

    def hdbscan_factory(
        min_cluster_size: int = 5,
        min_samples: int | None = None,
    ) -> Any:
        """Factory function for sklearn HDBSCAN when supported by the installed version."""
        if HDBSCAN is None:
            raise RuntimeError("HDBSCAN is unavailable in the installed scikit-learn version")
        return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

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
        dbscan_factory,
        {
            "type": "clustering",
            "algorithm": "DBSCAN",
            "description": "Density-based spatial clustering",
            "status": "active",
            "parameters": ["eps", "min_samples"],
        },
    )

    register(
        "hdbscan",
        hdbscan_factory,
        {
            "type": "clustering",
            "algorithm": "HDBSCAN",
            "description": "Hierarchical density-based clustering",
            "status": "active" if HDBSCAN is not None else "unavailable",
            "parameters": ["min_cluster_size", "min_samples"],
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
