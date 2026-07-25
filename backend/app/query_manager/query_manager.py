"""ASTER query manager module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.context_builder.context_builder import build_context, normalize_query
from backend.app.decision_engine.decision_engine import generate_explanations
from backend.app.execution_graph.execution_graph import build_execution_graph
from backend.app.planner.planner import build_execution_plan
from backend.app.response_composer.response_composer import compose_response
from backend.app.scheduler.scheduler import execute_graph


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


def execute_query(query: str, dataset_path: str | Path | None = None) -> dict[str, Any]:
    """Run the full analytical pipeline for a natural-language query."""

    context = build_context(query, dataset_path=dataset_path)
    planner_output = build_execution_plan(context)
    graph = build_execution_graph(planner_output)

    n_clusters = context.get("filters", {}).get("n_clusters", 3)
    initial_context: dict[str, Any] = {
        "dataset_path": Path(context["dataset_path"]),
        "n_clusters": n_clusters,
        "output_path": Path("backend/data/processed/customer_features.csv"),
        "query_context": context,
        "analytical_intent": planner_output.get("intent", context["intent"]),
    }

    node_outputs = execute_graph(graph, initial_context)
    explanations = generate_explanations(node_outputs)
    return compose_response(
        workflow_name=graph.workflow_name,
        intent=graph.intent,
        intent_classification=graph.intent_classification,
        node_outputs=node_outputs,
        explanations=explanations,
        execution_log=planner_output.get("execution_log", []),
        planning_path=planner_output.get("planning_path"),
        planner_reasoning=planner_output.get("planner_reasoning"),
    )
