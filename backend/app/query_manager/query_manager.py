"""ASTER query manager module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.app.context_builder.context_builder import build_context, normalize_query
from backend.app.decision_engine.decision_engine import generate_explanations
from backend.app.decision_memory.decision_memory import lookup as dm_lookup
from backend.app.decision_memory.decision_memory import store as dm_store
from backend.app.execution_graph.execution_graph import build_execution_graph
from backend.app.planner.planner import build_execution_plan
from backend.app.response_composer.response_composer import compose_response
from backend.app.scheduler.scheduler import execute_graph

logger = logging.getLogger(__name__)


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
    """Run the full analytical pipeline for a natural-language query.

    Before running the pipeline, checks decision memory for an exact-key
    cache hit. On hit: returns the stored response directly, skipping
    Planner/Scheduler/Nodes. On miss: runs normally, then writes.
    """

    # --- Decision memory cache read ---
    cached = dm_lookup(query)
    if cached is not None:
        logger.info("Decision memory cache hit for query: %s", query[:80])
        return cached

    # --- Full pipeline ---
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

    from backend.app.scheduler.scheduler import SchedulerExecutionError

    try:
        node_outputs = execute_graph(graph, initial_context)
    except SchedulerExecutionError as e:
        node_outputs = getattr(e, 'partial_outputs', {})
        node_summary = {
            node: {
                "type": type(output).__name__,
                "duration_ms": output.get("_duration_ms", 0) if isinstance(output, dict) else 0,
                "status": output.get("_status", "failed") if isinstance(output, dict) else "failed"
            }
            for node, output in node_outputs.items()
        }
        try:
            dm_store(
                query_text=query,
                response={"error": str(e)},
                execution_graph_summary=[node.to_dict() for node in graph.nodes],
                chosen_algorithm=None,
                node_outputs_summary=node_summary,
                explanation_summary=None
            )
        except Exception:
            pass
        raise

    explanations = generate_explanations(node_outputs)
    response = compose_response(
        workflow_name=graph.workflow_name,
        intent=graph.intent,
        intent_classification=graph.intent_classification,
        node_outputs=node_outputs,
        explanations=explanations,
        execution_log=planner_output.get("execution_log", []),
        planning_path=planner_output.get("planning_path"),
        planner_reasoning=planner_output.get("planner_reasoning"),
        unsupported_filters=context.get("unsupported_filters", []),
    )

    # --- Decision memory cache write (fire-and-forget) ---
    try:
        execution_graph_summary = [node.to_dict() for node in graph.nodes]
        chosen_algorithm = node_outputs.get("segmentation", {}).get("chosen_algorithm")
        node_summary = {
            node: {
                "type": type(output).__name__,
                "duration_ms": output.get("_duration_ms", 0) if isinstance(output, dict) else 0,
                "status": output.get("_status", "success") if isinstance(output, dict) else "success"
            }
            for node, output in node_outputs.items()
        }
        explanation_summary = {
            "explainer_used": explanations.get("explainer_used"),
            "explainer_reason": explanations.get("explainer_reason"),
            "customer_count": len(explanations.get("customer_explanations", [])),
            "segment_count": len(explanations.get("segment_summaries", [])),
        }
        dm_store(
            query_text=query,
            response=response,
            execution_graph_summary=execution_graph_summary,
            chosen_algorithm=chosen_algorithm,
            node_outputs_summary=node_summary,
            explanation_summary=explanation_summary,
        )
    except Exception:
        logger.exception("Decision memory write failed (fire-and-forget)")

    return response
