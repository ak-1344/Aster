"""ASTER planner module with bounded Gemini reasoning and deterministic fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.llm.gemini_client import (
    GeminiRequestError,
    GeminiResponseError,
    GeminiUnavailableError,
    request_structured_output,
)
from backend.utils.loader import DEFAULT_DATASET_PATH


VALID_PLAN_INTENTS = {"full_workflow", "explanation_only", "eda_only"}

# Logical tool names used in query-aware execution summaries.
TOOL_EDA = "EDA_Tool"
TOOL_FEATURE_ENGINEERING = "Feature_Engineering"
TOOL_SEGMENTATION = "Segmentation_Engine"
TOOL_RECOMMENDATION = "Recommendation_Engine"
TOOL_CUSTOMER_LOOKUP = "Single_Customer_Lookup"
TOOL_SEGMENT_EXPLAINER = "Segment_Rule_Explainer"

# Subjective language that requires an explicit operational definition.
SUBJECTIVE_TERMS = {
    "best",
    "profitable",
    "top customers",
    "top customer",
    "best customers",
    "best customer",
    "most valuable",
    "high value",
    "high-value",
}

_CUST_ID_PATTERN = re.compile(r"\bC\d{4,}\b", re.IGNORECASE)


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
        "purpose": "Provide exploratory summaries, distributions, and null checks",
        "inputs": ["backend/data/raw/CC GENERAL.csv"],
        "outputs": ["missing_values", "numeric_correlations", "sample_rows"],
        "tool": TOOL_EDA,
    },
    "feature_engineering": {
        "purpose": "Load and prepare customer-level features",
        "inputs": ["backend/data/raw/CC GENERAL.csv"],
        "outputs": ["backend/data/processed/customer_features.csv"],
        "tool": TOOL_FEATURE_ENGINEERING,
    },
    "segmentation": {
        "purpose": "Cluster customers into behavioural groups",
        "inputs": ["customer_features.csv"],
        "outputs": ["cluster_labels", "cluster_centers"],
        "tool": TOOL_SEGMENTATION,
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
        "tool": TOOL_RECOMMENDATION,
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
    explanation_keywords = {
        "explain",
        "why",
        "how",
        "meaning",
        "interpret",
        "understand",
        "clarify",
    }
    eda_keywords = {
        "explore",
        "analyze",
        "summary",
        "statistics",
        "describe",
        "overview",
        "profile",
    }

    tokens = set(query_text.split())
    has_explanation = bool(tokens & explanation_keywords)
    has_eda = bool(tokens & eda_keywords)

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
    """Feature_Engineering -> Segmentation_Engine -> Recommendation_Engine."""

    return _build_steps_from_sequence(
        ["feature_engineering", "segmentation", "recommendation"]
    )


def _build_descriptive_steps() -> list[PlanStep]:
    """Run ONLY EDA_Tool for descriptive intent."""

    return _build_steps_from_sequence(["eda"])


def extract_customer_id(query_text: str) -> str | None:
    """Extract a CUST_ID token (e.g. C10001) from free text when present."""

    match = _CUST_ID_PATTERN.search(query_text or "")
    return match.group(0).upper() if match else None


def select_explanation_tool(query_text: str) -> str:
    """Choose Single_Customer_Lookup vs Segment_Rule_Explainer for explanation intent."""

    if extract_customer_id(query_text):
        return TOOL_CUSTOMER_LOOKUP
    return TOOL_SEGMENT_EXPLAINER


def detect_subjective_assumption(query_text: str) -> str | None:
    """Return an operational definition when subjective language lacks thresholds."""

    normalized = (query_text or "").lower()
    if re.search(r"\d", normalized):
        return None

    hit = next((term for term in SUBJECTIVE_TERMS if term in normalized), None)
    if hit is None:
        return None

    return (
        f"Assumed '{hit}' refers to 'The Transactors' segment with high purchase "
        "volume and strong repayment behaviour (high full-payment tendency / low "
        "revolving risk), used as the operational definition of best customers."
    )


def resolve_tools_for_intent(
    analytical_intent: str,
    query_text: str,
    node_sequence: list[str] | None = None,
) -> list[str]:
    """Map analytical intent (and optional nodes) to logical tool names."""

    if analytical_intent == "descriptive":
        return [TOOL_EDA]
    if analytical_intent == "segmentation":
        if node_sequence:
            tools = [
                NODE_CATALOG[node]["tool"]
                for node in node_sequence
                if node in NODE_CATALOG and NODE_CATALOG[node].get("tool")
            ]
            return tools or [TOOL_FEATURE_ENGINEERING, TOOL_SEGMENTATION, TOOL_RECOMMENDATION]
        return [TOOL_FEATURE_ENGINEERING, TOOL_SEGMENTATION, TOOL_RECOMMENDATION]
    if analytical_intent == "explanation":
        return [select_explanation_tool(query_text)]
    return [TOOL_EDA]


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
            "tool": definition.get("tool"),
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
            "Intent orchestration rules:",
            "- descriptive / eda_only: schedule ONLY the eda node (EDA_Tool).",
            "- segmentation / full_workflow marketing queries: feature_engineering -> segmentation -> recommendation.",
            "- explanation_only: empty node_sequence; Single_Customer_Lookup or Segment_Rule_Explainer runs in answer formatting.",
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
    if intent == "eda_only" and (node_sequence != ["eda"]):
        raise InvalidLLMPlan("eda_only plans must schedule only the eda node")
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
    """Build a safe plan using intent-driven tool orchestration templates."""

    intent_classification = classify_intent(context)
    analytical_intent = str(context.get("intent", "descriptive"))

    if analytical_intent == "explanation" or intent_classification == "explanation_only":
        return "explanation_only", []
    if analytical_intent == "descriptive" or intent_classification == "eda_only":
        return "eda_only", _build_descriptive_steps()
    if analytical_intent == "segmentation":
        return "full_workflow", _build_segmentation_steps()
    return "eda_only", _build_descriptive_steps()


def _effective_analytical_intent(context: dict[str, Any], steps: list[PlanStep]) -> str:
    """Describe the workflow from its selected nodes rather than a keyword hint alone."""

    if any(step.node == "segmentation" for step in steps):
        return "segmentation"
    if any(step.node == "eda" for step in steps) and not any(
        step.node in {"feature_engineering", "segmentation", "recommendation"} for step in steps
    ):
        return "descriptive"
    context_intent = str(context.get("intent", "descriptive"))
    if context_intent == "explanation" and not steps:
        return "explanation"
    return context_intent


def build_execution_plan(context: dict[str, Any]) -> dict[str, Any]:
    """Build an executable plan with Gemini reasoning and a deterministic fallback."""

    query_text = str(
        context.get("normalized_query")
        or context.get("raw_query")
        or context.get("query", "")
    )
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
    node_sequence = [step.node for step in steps]
    tools_invoked = resolve_tools_for_intent(analytical_intent, query_text, node_sequence)
    operational_assumption = detect_subjective_assumption(query_text)

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
        "tools_invoked": tools_invoked,
        "operational_assumption": operational_assumption,
        "execution_log": [
            {
                "stage": "planner",
                "path": planning_path,
                "reason": planner_reasoning,
                "attempts": attempts,
                "tools_invoked": tools_invoked,
                "operational_assumption": operational_assumption,
            }
        ],
    }


def _infer_target_persona(query_text: str) -> str:
    """Map business phrasing to a persona from the segmentation tool catalog."""

    text = (query_text or "").lower()
    if any(token in text for token in ("cash advance", "overdraft", "liquidity")):
        return "Cash-Advance Reliant"
    if any(token in text for token in ("dormant", "inactive", "churn", "win back", "reactivat")):
        return "Dormant / Low Activity"
    if any(token in text for token in ("revolver", "debt", "balance transfer", "consolidat")):
        return "The Revolvers"
    # Premium / investment / best / top / default marketing target.
    return "The Transactors"


@lru_cache(maxsize=2)
def _cached_persona_frame(dataset_path: str) -> pd.DataFrame:
    """Cache persona-clustered frame for answer formatting."""

    from backend.app.tools.segmentation import preprocess_and_cluster

    result = preprocess_and_cluster(dataset_path=dataset_path)
    return result["dataframe"]


def _persona_frame(dataset_path: str | Path | None = None) -> pd.DataFrame:
    path = str(Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH)
    return _cached_persona_frame(path).copy()


def _markdown_customer_table(rows: pd.DataFrame) -> str:
    """Render the required CUST_ID marketing sample table."""

    columns = ["CUST_ID", "BALANCE", "PURCHASES", "CREDIT_LIMIT", "CREDIT_UTILIZATION"]
    present = [column for column in columns if column in rows.columns]
    if not present or rows.empty:
        return "_No matching customers available for this query._"

    view = rows.loc[:, present].copy()
    if "CREDIT_UTILIZATION" in view.columns:
        view["CREDIT_UTILIZATION"] = view["CREDIT_UTILIZATION"].map(lambda value: f"{float(value):.3f}")
    for numeric_col in ("BALANCE", "PURCHASES", "CREDIT_LIMIT"):
        if numeric_col in view.columns:
            view[numeric_col] = view[numeric_col].map(lambda value: f"{float(value):,.2f}")

    header = "| " + " | ".join(present) + " |"
    separator = "| " + " | ".join("---" for _ in present) + " |"
    body = [
        "| " + " | ".join(str(view.iloc[index][column]) for column in present) + " |"
        for index in range(len(view))
    ]
    return "\n".join([header, separator, *body])


def _strategic_recommendation(persona: str, product: str) -> str:
    """Persona-specific marketing guidance for the formatted answer."""

    playbooks = {
        "The Transactors": (
            f"Prioritize premium acquisition messaging for '{persona}'. Lead with "
            f"{product}, highlight travel/lifestyle rewards, and invite cardholders "
            "with strong repayment behaviour into a concierge upgrade path."
        ),
        "The Revolvers": (
            f"For '{persona}', lead with debt relief: position {product} as a lower-cost "
            "exit from revolving balances, pair with autopay enrollment, and suppress "
            "high-APR cash offers until utilization declines."
        ),
        "Cash-Advance Reliant": (
            f"Engage '{persona}' with liquidity alternatives. Offer {product}, "
            "emphasize transparent fees, and route customers away from repeated cash advances."
        ),
        "Dormant / Low Activity": (
            f"Reactivate '{persona}' with low-friction nudges. Promote {product}, "
            "use first-purchase cashback, and keep onboarding steps minimal."
        ),
    }
    return playbooks.get(
        persona,
        f"Target the '{persona}' cohort with {product} using behaviour-aligned messaging.",
    )


def format_high_impact_answer(
    context: dict[str, Any],
    planner_output: dict[str, Any],
    node_outputs: dict[str, Any] | None = None,
    explanations: dict[str, Any] | None = None,
    top_n: int = 8,
) -> dict[str, Any]:
    """Format the final agent answer into the required four-section structure."""

    from backend.app.tools.segmentation import (
        PERSONA_CATALOG,
        get_customer_insight,
        get_top_customers_for_persona,
    )

    node_outputs = node_outputs or {}
    explanations = explanations or {}
    query_text = str(context.get("raw_query") or context.get("normalized_query") or "")
    analytical_intent = str(planner_output.get("intent") or context.get("intent") or "descriptive")
    planning_path = planner_output.get("planning_path", "rule_based_fallback")
    tools_invoked = list(
        planner_output.get("tools_invoked")
        or resolve_tools_for_intent(analytical_intent, query_text)
    )
    operational_assumption = planner_output.get("operational_assumption") or detect_subjective_assumption(
        query_text
    )

    dataset_path = context.get("dataset_path")
    persona = _infer_target_persona(query_text)
    persona_meta = next(
        (details for details in PERSONA_CATALOG.values() if details["persona"] == persona),
        PERSONA_CATALOG[0],
    )
    product = persona_meta["product"]
    sample_rows = pd.DataFrame()
    primary_finding = ""

    if analytical_intent == "descriptive":
        eda = node_outputs.get("eda") or {}
        missing = eda.get("missing_values") or {}
        top_missing = sorted(missing.items(), key=lambda item: item[1], reverse=True)[:3]
        missing_bits = ", ".join(f"{column}={count}" for column, count in top_missing) or "none material"
        primary_finding = (
            f"EDA complete on {eda.get('row_count', 'the')} rows / "
            f"{eda.get('column_count', 'n')} columns. Notable null concentrations: {missing_bits}."
        )
        sample_rows = pd.DataFrame(eda.get("sample_rows") or [])
    elif analytical_intent == "explanation":
        frame = _persona_frame(dataset_path)
        cust_id = extract_customer_id(query_text)
        if cust_id:
            insight = get_customer_insight(cust_id, frame)
            persona = str(insight.get("persona") or persona)
            product = str(insight.get("recommended_product") or product)
            primary_finding = (
                f"Customer {cust_id} maps to '{persona}' "
                f"({insight.get('persona_profile')}). Recommended product: {product}."
            )
            sample_rows = frame[frame["CUST_ID"].astype(str) == cust_id]
        else:
            primary_finding = (
                f"Segment rules center on behavioural personas. "
                f"For this query, '{persona}' ({persona_meta['profile']}) is the matched rule set; "
                f"recommended offer: {product}."
            )
            sample_rows = get_top_customers_for_persona(persona, frame, top_n=top_n)
    else:
        frame = _persona_frame(dataset_path)
        sample_rows = get_top_customers_for_persona(persona, frame, top_n=top_n)
        primary_finding = (
            f"To address this request, target the '{persona}' persona "
            f"({persona_meta['profile']}). Recommended product: {product}."
        )

    if operational_assumption:
        primary_finding = f"{operational_assumption} {primary_finding}".strip()

    execution_bullets = [
        f"- Tools invoked: {', '.join(tools_invoked) if tools_invoked else 'none'}",
        f"- Planning path: `{planning_path}`",
        f"- Workflow: `{planner_output.get('workflow_name', analytical_intent + '_workflow')}`",
    ]
    if node_outputs:
        execution_bullets.append(
            f"- Nodes executed: {', '.join(node_outputs.keys())}"
        )
    if explanations.get("explainer_used"):
        execution_bullets.append(
            f"- Explainer: {explanations.get('explainer_used')} "
            f"({explanations.get('explainer_reason', '')})"
        )

    strategic = _strategic_recommendation(persona, product)
    table_markdown = _markdown_customer_table(sample_rows.head(top_n))

    markdown = "\n\n".join(
        [
            "## 1. Query-Aware Execution Summary\n" + "\n".join(execution_bullets),
            "## 2. Primary Finding & Persona Match\n" + primary_finding,
            "## 3. Target Customer Table\n" + table_markdown,
            "## 4. Strategic Marketing Recommendation\n" + strategic,
        ]
    )

    table_records = []
    if not sample_rows.empty:
        export_cols = [
            column
            for column in ["CUST_ID", "BALANCE", "PURCHASES", "CREDIT_LIMIT", "CREDIT_UTILIZATION"]
            if column in sample_rows.columns
        ]
        table_records = sample_rows.loc[:, export_cols].head(top_n).to_dict(orient="records")

    return {
        "markdown": markdown,
        "execution_summary": execution_bullets,
        "primary_finding": primary_finding,
        "target_customers": table_records,
        "strategic_recommendation": strategic,
        "persona": persona,
        "recommended_product": product,
        "tools_invoked": tools_invoked,
        "operational_assumption": operational_assumption,
        "planning_path": planning_path,
    }
