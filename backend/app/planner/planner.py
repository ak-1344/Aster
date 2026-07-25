"""ASTER planner module with bounded Gemini reasoning and deterministic fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from backend.app.llm.gemini_client import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiUnavailableError,
    request_structured_output,
)


VALID_PLAN_INTENTS = {"full_workflow", "explanation_only", "eda_only"}


class InvalidLLMPlan(ValueError):
    """Raised when an LLM plan is not safe to execute."""


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


NODE_CATALOG: dict[str, dict[str, Any]] = {
    "analytics": {
        "purpose": "Summarize dataset statistics",
        "inputs": ["backend/data/raw/CC GENERAL.csv"],
        "outputs": ["numeric_summary", "categorical_summary"],
    },
    "eda": {
        "purpose": "Provide exploratory summaries and correlations",
        "inputs": ["backend/data/raw/CC GENERAL.csv"],
        "outputs": ["missing_values", "numeric_correlations", "sample_rows"],
    },
    "feature_engineering": {
        "purpose": "Load and prepare customer-level features",
        "inputs": ["backend/data/raw/CC GENERAL.csv"],
        "outputs": ["backend/data/processed/customer_features.csv"],
    },
    "segmentation": {
        "purpose": "Cluster customers into behavioural groups",
        "inputs": ["customer_features.csv"],
        "outputs": ["cluster_labels", "cluster_centers"],
    },
    "evaluation": {
        "purpose": "Score the clustering quality",
        "inputs": ["cluster_labels", "customer_features.csv"],
        "outputs": ["silhouette_score", "cluster_sizes"],
    },
    "recommendation": {
        "purpose": "Attach tier-based retention and cross-sell guidance",
        "inputs": ["cluster_labels", "customer_features.csv"],
        "outputs": ["cluster_recommendations", "customer_recommendations"],
    },
    "visualization": {
        "purpose": "Prepare cluster charts for display",
        "inputs": ["cluster_labels", "customer_features.csv"],
        "outputs": ["scatter", "cluster_size_bar"],
    },
}

NODE_PREREQUISITES: dict[str, set[str]] = {
    "analytics": set(),
    "eda": set(),
    "feature_engineering": set(),
    "segmentation": {"feature_engineering"},
    "evaluation": {"feature_engineering", "segmentation"},
    "recommendation": {"feature_engineering", "segmentation"},
    "visualization": {"feature_engineering", "segmentation"},
}


def classify_intent(query_context: dict[str, Any]) -> str:
    """Classify the query with the original rule-based fallback logic."""

    query_text = str(
        query_context.get("normalized_query")
        or query_context.get("raw_query")
        or query_context.get("query", "")
    ).lower()
    explanation_keywords = [
        "explain",
        "why",
        "how",
        "meaning",
        "interpret",
        "understand",
        "clarify",
    ]
    eda_keywords = [
        "explore",
        "analyze",
        "summary",
        "statistics",
        "describe",
        "overview",
        "profile",
    ]

    has_explanation = any(keyword in query_text for keyword in explanation_keywords)
    has_eda = any(keyword in query_text for keyword in eda_keywords)

    if has_explanation and not has_eda:
        return "explanation_only"
    if has_eda and not has_explanation:
        return "eda_only"
    return "full_workflow"


def _build_steps_from_sequence(node_sequence: list[str]) -> list[PlanStep]:
    """Create concrete plan steps from a validated node sequence."""

    return [
        PlanStep(
            node=node_name,
            purpose=NODE_CATALOG[node_name]["purpose"],
            inputs=list(NODE_CATALOG[node_name]["inputs"]),
            outputs=list(NODE_CATALOG[node_name]["outputs"]),
        )
        for node_name in node_sequence
    ]


def _build_segmentation_steps() -> list[PlanStep]:
    """Build the original segmentation template for rule-based fallback only."""

    return _build_steps_from_sequence(
        [
            "feature_engineering",
            "segmentation",
            "evaluation",
            "recommendation",
            "visualization",
        ]
    )


def _build_descriptive_steps() -> list[PlanStep]:
    """Build the original descriptive template for rule-based fallback only."""

    return _build_steps_from_sequence(["analytics", "eda"])


def _plan_schema() -> dict[str, Any]:
    """Return the Gemini JSON schema for a planner response."""

    return {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": sorted(VALID_PLAN_INTENTS),
            },
            "node_sequence": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(NODE_CATALOG)},
            },
            "reasoning": {
                "type": "string",
                "description": "A brief user-facing reason for the selected nodes.",
            },
        },
        "required": ["intent", "node_sequence", "reasoning"],
        "additionalProperties": False,
    }


def _planner_prompt(context: dict[str, Any], retry_feedback: str = "") -> str:
    """Build a constrained planner prompt from the normalized query context."""

    catalog = {
        node_name: {
            "purpose": definition["purpose"],
            "must_follow": sorted(NODE_PREREQUISITES[node_name]),
        }
        for node_name, definition in NODE_CATALOG.items()
    }
    context_for_prompt = {
        "normalized_query": context.get("normalized_query", ""),
        "intent_hint": context.get("intent", "descriptive"),
        "entities": context.get("entities", []),
        "filters": context.get("filters", {}),
        "output_format": context.get("output_format", "table"),
    }

    return "\n".join(
        [
            "You are the ASTER analytical workflow planner.",
            "Choose the smallest safe ordered node sequence that answers the request.",
            "Return only a JSON object matching the supplied schema.",
            "Do not invent node names, algorithms, data, recommendations, or computations.",
            "The recommendation node is only needed when the user asks for actions, offers, or recommendations.",
            "The visualization node currently renders customer clusters, so it requires feature_engineering and segmentation first.",
            "Use explanation_only with an empty node_sequence only when no new analytical execution is needed.",
            f"Query context: {json.dumps(context_for_prompt, sort_keys=True)}",
            f"Node catalog: {json.dumps(catalog, sort_keys=True)}",
            retry_feedback,
        ]
    )


def _validate_node_sequence(intent: str, node_sequence: list[Any]) -> list[str]:
    """Ensure an LLM-proposed sequence contains only executable known nodes."""

    if not isinstance(node_sequence, list) or not all(
        isinstance(node_name, str) for node_name in node_sequence
    ):
        raise InvalidLLMPlan("node_sequence must be a list of node names")

    if len(node_sequence) != len(set(node_sequence)):
        raise InvalidLLMPlan("node_sequence cannot contain duplicate nodes")

    invalid_nodes = set(node_sequence) - set(NODE_CATALOG)
    if invalid_nodes:
        raise InvalidLLMPlan(
            f"node_sequence contains unknown nodes: {', '.join(sorted(invalid_nodes))}"
        )

    if intent == "explanation_only" and node_sequence:
        raise InvalidLLMPlan("explanation_only plans must not schedule analytical nodes")
    if intent == "eda_only" and (
        not node_sequence or any(node not in {"analytics", "eda"} for node in node_sequence)
    ):
        raise InvalidLLMPlan("eda_only plans may contain only analytics and eda nodes")
    if intent == "full_workflow" and not node_sequence:
        raise InvalidLLMPlan("full_workflow plans require at least one node")

    completed: set[str] = set()
    for node_name in node_sequence:
        missing = NODE_PREREQUISITES[node_name] - completed
        if missing:
            raise InvalidLLMPlan(
                f"{node_name} is missing prior dependencies: {', '.join(sorted(missing))}"
            )
        completed.add(node_name)

    return list(node_sequence)


def _validate_llm_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the typed shape and executable constraints of a Gemini plan."""

    intent = payload.get("intent")
    if intent not in VALID_PLAN_INTENTS:
        raise InvalidLLMPlan("intent is not an allowed planner intent")

    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise InvalidLLMPlan("reasoning must be a non-empty string")

    return {
        "intent": intent,
        "node_sequence": _validate_node_sequence(intent, payload.get("node_sequence")),
        "reasoning": reasoning.strip(),
    }


def _request_llm_plan(context: dict[str, Any]) -> tuple[dict[str, Any] | None, str, int]:
    """Request and validate a Gemini plan, retrying once for invalid output."""

    retry_feedback = ""
    for attempt in range(1, 3):
        try:
            payload = request_structured_output(
                _planner_prompt(context, retry_feedback),
                _plan_schema(),
            )
            return _validate_llm_plan(payload), "", attempt
        except (GeminiUnavailableError, GeminiRequestError) as error:
            return None, str(error), attempt
        except (GeminiResponseError, InvalidLLMPlan) as error:
            if attempt == 2:
                return None, f"Gemini output remained invalid after retry: {error}", attempt
            retry_feedback = (
                "Your previous response was rejected because "
                f"{error}. Return a corrected JSON object using only the supplied catalog."
            )

    return None, "Gemini planning did not produce a usable response", 2


def _build_rule_based_fallback(context: dict[str, Any]) -> tuple[str, list[PlanStep]]:
    """Build a safe plan using the preserved rule-based classifier and templates."""

    intent_classification = classify_intent(context)
    analytical_intent = context.get("intent", "descriptive")

    if intent_classification == "explanation_only":
        return intent_classification, []
    if intent_classification == "eda_only":
        return intent_classification, _build_descriptive_steps()
    if analytical_intent == "segmentation":
        return intent_classification, _build_segmentation_steps()
    return intent_classification, _build_descriptive_steps()


def _effective_analytical_intent(context: dict[str, Any], steps: list[PlanStep]) -> str:
    """Describe the workflow from its selected nodes rather than a keyword hint alone."""

    if any(step.node == "segmentation" for step in steps):
        return "segmentation"
    return str(context.get("intent", "descriptive"))


def build_execution_plan(context: dict[str, Any]) -> dict[str, Any]:
    """Build an executable plan with Gemini reasoning and a deterministic fallback."""

    llm_plan, fallback_reason, attempts = _request_llm_plan(context)
    if llm_plan is not None:
        steps = _build_steps_from_sequence(llm_plan["node_sequence"])
        intent_classification = llm_plan["intent"]
        planning_path = "llm_reasoned"
        planner_reasoning = llm_plan["reasoning"]
    else:
        intent_classification, steps = _build_rule_based_fallback(context)
        planning_path = "rule_based_fallback"
        planner_reasoning = fallback_reason

    analytical_intent = _effective_analytical_intent(context, steps)
    return {
        "intent": analytical_intent,
        "context_intent": context.get("intent", "descriptive"),
        "intent_classification": intent_classification,
        "executable": True,
        "workflow_name": f"{analytical_intent}_workflow",
        "step_count": len(steps),
        "steps": [step.to_dict() for step in steps],
        "entrypoint": steps[0].node if steps else None,
        "exitpoint": steps[-1].node if steps else None,
        "planning_path": planning_path,
        "planner_reasoning": planner_reasoning,
        "execution_log": [
            {
                "stage": "planner",
                "path": planning_path,
                "reason": planner_reasoning,
                "attempts": attempts,
            }
        ],
    }
