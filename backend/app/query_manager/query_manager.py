"""ASTER query manager module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.context_builder.context_builder import build_context, normalize_query
from backend.app.planner.planner import build_execution_plan


def route_intent(query: str) -> str:
	"""Route the request to a high-level analytical intent."""

	normalized_query = normalize_query(query)
	context = build_context(query)
	if context["intent"] == "segmentation":
		return "segmentation"
	if context["intent"] == "descriptive":
		return "descriptive"
	if "recommend" in normalized_query:
		return "recommendation"
	return "descriptive"


def orchestrate_query(query: str, dataset_path: str | Path | None = None) -> dict[str, Any]:
	"""Build context and hand it to the planner."""

	context = build_context(query, dataset_path=dataset_path)
	plan = build_execution_plan(context)
	return {
		"context": context,
		"intent": context["intent"],
		"plan": plan,
		"executable": bool(plan.get("executable", False)),
	}