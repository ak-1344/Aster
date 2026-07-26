"""Tests for dynamic tool orchestration and high-impact answer formatting."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from backend.app.context_builder.context_builder import build_context
from backend.app.planner.planner import (
    TOOL_CUSTOMER_LOOKUP,
    TOOL_EDA,
    TOOL_FEATURE_ENGINEERING,
    TOOL_RECOMMENDATION,
    TOOL_SEGMENT_EXPLAINER,
    TOOL_SEGMENTATION,
    _build_rule_based_fallback,
    build_execution_plan,
    detect_subjective_assumption,
    format_high_impact_answer,
    select_explanation_tool,
)


class PlannerOrchestrationTests(unittest.TestCase):
    def test_descriptive_fallback_runs_eda_only(self) -> None:
        context = build_context("show descriptive statistics for the dataset")
        intent_classification, steps = _build_rule_based_fallback(context)

        self.assertEqual(intent_classification, "eda_only")
        self.assertEqual([step.node for step in steps], ["eda"])

    def test_segmentation_fallback_tool_chain(self) -> None:
        context = build_context("Find customers suitable for investment products")
        intent_classification, steps = _build_rule_based_fallback(context)

        self.assertEqual(intent_classification, "full_workflow")
        self.assertEqual(
            [step.node for step in steps],
            ["feature_engineering", "segmentation", "recommendation"],
        )

    def test_explanation_tool_selection(self) -> None:
        self.assertEqual(
            select_explanation_tool("explain why customer C10001 is in this segment"),
            TOOL_CUSTOMER_LOOKUP,
        )
        self.assertEqual(
            select_explanation_tool("explain why this segment exists"),
            TOOL_SEGMENT_EXPLAINER,
        )

    def test_subjective_assumption_without_threshold(self) -> None:
        assumption = detect_subjective_assumption("who are the best customers")
        self.assertIsNotNone(assumption)
        self.assertIn("The Transactors", assumption)

        self.assertIsNone(detect_subjective_assumption("customers with balance above 5000"))

    def test_plan_includes_tools_invoked(self) -> None:
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "INVALID_KEY_FOR_TESTING"}):
            plan = build_execution_plan(build_context("segment customers into 3 clusters"))

        self.assertEqual(plan["planning_path"], "rule_based_fallback")
        self.assertEqual(
            plan["tools_invoked"],
            [TOOL_FEATURE_ENGINEERING, TOOL_SEGMENTATION, TOOL_RECOMMENDATION],
        )

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "INVALID_KEY_FOR_TESTING"}):
            descriptive = build_execution_plan(
                build_context("show descriptive statistics for the dataset")
            )
        self.assertEqual(descriptive["tools_invoked"], [TOOL_EDA])

    def test_format_high_impact_answer_has_four_sections(self) -> None:
        context = build_context("Find customers suitable for investment products")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "INVALID_KEY_FOR_TESTING"}):
            plan = build_execution_plan(context)

        answer = format_high_impact_answer(context, plan, node_outputs={})
        markdown = answer["markdown"]

        self.assertIn("Query-Aware Execution Summary", markdown)
        self.assertIn("Primary Finding & Persona Match", markdown)
        self.assertIn("Target Customer Table", markdown)
        self.assertIn("Strategic Marketing Recommendation", markdown)
        self.assertIn("CUST_ID", markdown)
        self.assertIn("CREDIT_UTILIZATION", markdown)
        self.assertEqual(answer["persona"], "The Transactors")
        self.assertIn("The Transactors", answer["primary_finding"])


if __name__ == "__main__":
    unittest.main()
