"""Execution engine smoke tests for ASTER."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.context_builder.context_builder import build_context
from backend.app.execution_graph.execution_graph import build_execution_graph, ExecutionGraph
from backend.app.planner.planner import build_execution_plan
from backend.app.response_composer.response_composer import compose_response
from backend.app.scheduler.scheduler import execute_graph


class ExecutionEngineSmokeTests(unittest.TestCase):
    def test_execution_graph_builds_from_planner_output(self) -> None:
        """Test that execution graph can be built from planner output."""
        context = build_context("segment customers into 3 clusters")
        planner_output = build_execution_plan(context)
        graph = build_execution_graph(planner_output)
        
        self.assertIsInstance(graph, ExecutionGraph)
        self.assertEqual(graph.workflow_name, "segmentation_workflow")
        self.assertEqual(graph.intent, "segmentation")
        self.assertGreater(len(graph.nodes), 0)
        self.assertEqual(graph.entrypoint, "feature_engineering")
        self.assertEqual(graph.exitpoint, "recommendation")

    def test_execution_graph_gets_execution_order(self) -> None:
        """Test that execution graph returns nodes in dependency order."""
        context = build_context("segment customers into 3 clusters")
        planner_output = build_execution_plan(context)
        graph = build_execution_graph(planner_output)
        
        execution_order = graph.get_execution_order()
        
        self.assertIsInstance(execution_order, list)
        self.assertEqual(len(execution_order), len(graph.nodes))
        # feature_engineering should come before segmentation
        self.assertIn("feature_engineering", execution_order)
        self.assertIn("segmentation", execution_order)
        fe_idx = execution_order.index("feature_engineering")
        seg_idx = execution_order.index("segmentation")
        self.assertLess(fe_idx, seg_idx)

    def test_execution_graph_handles_descriptive_workflow(self) -> None:
        """Test that execution graph handles descriptive/EDA workflows."""
        context = build_context("show descriptive statistics for the dataset")
        planner_output = build_execution_plan(context)
        graph = build_execution_graph(planner_output)
        
        self.assertEqual(graph.workflow_name, "descriptive_workflow")
        self.assertEqual(graph.intent, "descriptive")
        execution_order = graph.get_execution_order()
        self.assertEqual(execution_order, ["eda"])

    def test_scheduler_executes_segmentation_workflow(self) -> None:
        """Test that scheduler can execute a segmentation workflow end-to-end."""
        context = build_context("segment customers into 3 clusters")
        planner_output = build_execution_plan(context)
        graph = build_execution_graph(planner_output)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "customer_features.csv"
            initial_context = {
                "dataset_path": Path("backend/data/raw/CC GENERAL.csv"),
                "n_clusters": 3,
                "output_path": output_path,
            }
            
            node_outputs = execute_graph(graph, initial_context)
            
            self.assertIn("feature_engineering", node_outputs)
            self.assertIn("segmentation", node_outputs)
            self.assertIn("recommendation", node_outputs)
            self.assertNotIn("evaluation", node_outputs)
            self.assertNotIn("visualization", node_outputs)
            
            # Verify segmentation output
            self.assertIn("labels", node_outputs["segmentation"])
            self.assertEqual(len(node_outputs["segmentation"]["labels"]), node_outputs["feature_engineering"]["row_count"])

    def test_scheduler_executes_descriptive_workflow(self) -> None:
        """Test that scheduler can execute a descriptive workflow end-to-end."""
        context = build_context("show descriptive statistics for the dataset")
        planner_output = build_execution_plan(context)
        graph = build_execution_graph(planner_output)
        
        initial_context = {
            "dataset_path": Path("backend/data/raw/CC GENERAL.csv"),
        }
        
        node_outputs = execute_graph(graph, initial_context)
        
        self.assertEqual(list(node_outputs.keys()), ["eda"])
        
        # Verify EDA output
        self.assertIn("numeric_correlations", node_outputs["eda"])
        self.assertIn("sample_rows", node_outputs["eda"])
        self.assertIn("missing_values", node_outputs["eda"])

    def test_response_composer_merges_segmentation_outputs(self) -> None:
        """Test that response composer merges segmentation workflow outputs."""
        context = build_context("segment customers into 3 clusters")
        planner_output = build_execution_plan(context)
        graph = build_execution_graph(planner_output)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "customer_features.csv"
            initial_context = {
                "dataset_path": Path("backend/data/raw/CC GENERAL.csv"),
                "n_clusters": 3,
                "output_path": output_path,
            }
            
            node_outputs = execute_graph(graph, initial_context)
            response = compose_response(
                workflow_name=graph.workflow_name,
                intent=graph.intent,
                intent_classification=graph.intent_classification,
                node_outputs=node_outputs,
            )
            
            self.assertEqual(response["workflow_name"], "segmentation_workflow")
            self.assertEqual(response["intent"], "segmentation")
            self.assertIn("summary", response)
            self.assertIn("statistics", response)
            self.assertIn("recommendations", response)
            self.assertIn("agent_answer", response)
            
            # Verify segmentation summary
            self.assertIn("segmentation", response["summary"])
            self.assertIn("cluster_count", response["summary"]["segmentation"])
            
            # Verify recommendations
            self.assertIn("cluster_recommendations", response["recommendations"])
            self.assertIn("customer_recommendations", response["recommendations"])
            
            # Verify high-impact answer structure
            self.assertIn("Primary Finding", response["agent_answer"].get("markdown", ""))
            self.assertIn("Target Customer Table", response["agent_answer"].get("markdown", ""))

    def test_response_composer_merges_descriptive_outputs(self) -> None:
        """Test that response composer merges descriptive workflow outputs."""
        context = build_context("show descriptive statistics for the dataset")
        planner_output = build_execution_plan(context)
        graph = build_execution_graph(planner_output)
        
        initial_context = {
            "dataset_path": Path("backend/data/raw/CC GENERAL.csv"),
        }
        
        node_outputs = execute_graph(graph, initial_context)
        response = compose_response(
            workflow_name=graph.workflow_name,
            intent=graph.intent,
            intent_classification=graph.intent_classification,
            node_outputs=node_outputs,
        )
        
        self.assertEqual(response["workflow_name"], "descriptive_workflow")
        self.assertEqual(response["intent"], "descriptive")
        self.assertIn("statistics", response)
        
        # Verify exploratory statistics from EDA_Tool only
        self.assertIn("exploratory", response["statistics"])
        self.assertIn("numeric_correlations", response["statistics"]["exploratory"])
        self.assertNotIn("descriptive", response["statistics"])

    def test_end_to_end_segmentation_workflow(self) -> None:
        """Test full end-to-end workflow: planner -> execution_graph -> scheduler -> response_composer."""
        context = build_context("segment customers into 3 clusters")
        planner_output = build_execution_plan(context)
        graph = build_execution_graph(planner_output)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "customer_features.csv"
            initial_context = {
                "dataset_path": Path("backend/data/raw/CC GENERAL.csv"),
                "n_clusters": 3,
                "output_path": output_path,
            }
            
            node_outputs = execute_graph(graph, initial_context)
            response = compose_response(
                workflow_name=graph.workflow_name,
                intent=graph.intent,
                intent_classification=graph.intent_classification,
                node_outputs=node_outputs,
            )
            
            # Verify complete response structure
            self.assertIn("workflow_name", response)
            self.assertIn("intent", response)
            self.assertIn("intent_classification", response)
            self.assertIn("summary", response)
            self.assertIn("statistics", response)
            self.assertIn("recommendations", response)
            self.assertIn("visualizations", response)
            self.assertIn("metadata", response)
            
            # Verify metadata
            self.assertEqual(len(response["metadata"]["nodes_executed"]), 3)
            self.assertIn("feature_engineering", response["metadata"]["nodes_executed"])
            self.assertIn("segmentation", response["metadata"]["nodes_executed"])
            self.assertIn("recommendation", response["metadata"]["nodes_executed"])

    def test_end_to_end_descriptive_workflow(self) -> None:
        """Test full end-to-end workflow for descriptive query."""
        context = build_context("show descriptive statistics for the dataset")
        planner_output = build_execution_plan(context)
        graph = build_execution_graph(planner_output)
        
        initial_context = {
            "dataset_path": Path("backend/data/raw/CC GENERAL.csv"),
        }
        
        node_outputs = execute_graph(graph, initial_context)
        response = compose_response(
            workflow_name=graph.workflow_name,
            intent=graph.intent,
            intent_classification=graph.intent_classification,
            node_outputs=node_outputs,
        )
        
        # Verify complete response structure
        self.assertEqual(response["workflow_name"], "descriptive_workflow")
        self.assertEqual(response["intent"], "descriptive")
        self.assertIn("statistics", response)
        self.assertIn("exploratory", response["statistics"])
        self.assertEqual(response["metadata"]["nodes_executed"], ["eda"])


if __name__ == "__main__":
    unittest.main()
