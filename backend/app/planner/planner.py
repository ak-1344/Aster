"""ASTER planner module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlanStep:
	"""A single execution step in the planned workflow."""

	node: str
	purpose: str
	inputs: list[str] = field(default_factory=list)
	outputs: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		"""Return a JSON-serializable representation of the step."""

		return {
			"node": self.node,
			"purpose": self.purpose,
			"inputs": self.inputs,
			"outputs": self.outputs,
		}


def _build_segmentation_steps(n_clusters: int) -> list[PlanStep]:
	"""Build the current segmentation workflow steps."""

	return [
		PlanStep(
			node="feature_engineering",
			purpose="Load and prepare customer-level features",
			inputs=["backend/data/raw/CC GENERAL.csv"],
			outputs=["backend/data/processed/customer_features.csv"],
		),
		PlanStep(
			node="segmentation",
			purpose="Cluster customers into behavioural groups",
			inputs=["customer_features.csv"],
			outputs=["cluster_labels", "cluster_centers"],
		),
		PlanStep(
			node="evaluation",
			purpose="Score the clustering quality",
			inputs=["cluster_labels", "customer_features.csv"],
			outputs=["silhouette_score", "cluster_sizes"],
		),
		PlanStep(
			node="recommendation",
			purpose="Attach tier-based retention and cross-sell guidance",
			inputs=["cluster_labels", "customer_features.csv"],
			outputs=["cluster_recommendations", "customer_recommendations"],
		),
		PlanStep(
			node="visualization",
			purpose="Prepare cluster charts for display",
			inputs=["cluster_labels", "customer_features.csv"],
			outputs=["scatter", "cluster_size_bar"],
		),
	]


def _build_descriptive_steps() -> list[PlanStep]:
	"""Build the current descriptive-analysis workflow steps."""

	return [
		PlanStep(
			node="analytics",
			purpose="Summarize dataset statistics",
			inputs=["backend/data/raw/CC GENERAL.csv"],
			outputs=["numeric_summary", "categorical_summary"],
		),
		PlanStep(
			node="eda",
			purpose="Provide exploratory summaries and correlations",
			inputs=["backend/data/raw/CC GENERAL.csv"],
			outputs=["missing_values", "numeric_correlations", "sample_rows"],
		),
	]


def build_execution_plan(context: dict[str, Any]) -> dict[str, Any]:
	"""Build an executable workflow for the current analytical request."""

	intent = context.get("intent", "descriptive")
	filters = context.get("filters", {})

	if intent == "segmentation":
		steps = _build_segmentation_steps(int(filters.get("n_clusters", 3)))
	else:
		steps = _build_descriptive_steps()

	return {
		"intent": intent,
		"executable": True,
		"workflow_name": f"{intent}_workflow",
		"step_count": len(steps),
		"steps": [step.to_dict() for step in steps],
		"entrypoint": steps[0].node if steps else None,
		"exitpoint": steps[-1].node if steps else None,
	}