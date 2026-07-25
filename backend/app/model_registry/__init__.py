"""ASTER model registry module."""

from __future__ import annotations

from backend.app.model_registry.model_registry import (
    get,
    list_available,
    register,
    registry,
)

__all__ = ["registry", "register", "get", "list_available"]
