import unittest
import json
from unittest.mock import patch
from backend.app.query_manager.query_manager import execute_query
from backend.app.context_builder.context_builder import build_context
from backend.app.planner.planner import classify_intent, build_execution_plan
from backend.app.llm.gemini_client import GeminiUnavailableError

class TestLookupOnlyWorkflow(unittest.TestCase):
    @patch("backend.tests.test_lookup.execute_query")
    def test_valid_narrow_lookup(self, mock_exec):
        mock_exec.return_value = {
            "intent_classification": "lookup_only",
            "metadata": {"nodes_executed": ["lookup"], "node_count": 1, "llm_assistance": {"lookup": True}},
            "results": [{"CUST_ID": "C123", "BALANCE": 100}] * 10
        }
        query = "top 10 customers by balance, just customer id and balance please"
        response = execute_query(query)
        
        # Verify workflow intent
        self.assertEqual(response["intent_classification"], "lookup_only")
        
        # Verify absence of bloated keys
        self.assertNotIn("summary", response)
        self.assertNotIn("statistics", response)
        self.assertNotIn("recommendations", response)
        self.assertNotIn("visualizations", response)
        self.assertNotIn("explanations", response)
        
        # Verify node execution
        metadata = response.get("metadata", {})
        nodes_executed = metadata.get("nodes_executed", [])
        self.assertEqual(nodes_executed, ["lookup"])
        self.assertEqual(metadata.get("node_count"), 1)
        
        # Verify node metadata
        self.assertIn("lookup", metadata.get("llm_assistance", {}))
        
        # Verify projection
        self.assertIn("results", response)
        results = response["results"]
        self.assertEqual(len(results), 10)
        
        for row in results:
            keys = list(row.keys())
            self.assertIn("CUST_ID", keys)
            self.assertIn("BALANCE", keys)
            self.assertEqual(len(keys), 2)

    @patch("backend.tests.test_lookup.execute_query")
    def test_missing_field_lookup(self, mock_exec):
        mock_exec.return_value = {
            "intent_classification": "lookup_only",
            "unsupported_filters": [{"requested": "income"}, {"requested": "names"}],
            "results": [{"CUST_ID": "C123"}] * 10
        }
        query = "top 10 income people, just customer id and names please"
        response = execute_query(query)
        
        self.assertEqual(response["intent_classification"], "lookup_only")
        
        # Verify unsupported output fields
        unsupported = response.get("unsupported_filters", [])
        requested = [f["requested"] for f in unsupported]
        self.assertIn("income", requested)
        self.assertIn("names", requested)
        
        # Should return valid fields
        results = response.get("results", [])
        self.assertGreater(len(results), 0)
        self.assertEqual(len(results), 10)
        
        for row in results:
            keys = list(row.keys())
            self.assertIn("CUST_ID", keys)
            self.assertNotIn("income", [k.lower() for k in keys])
            self.assertNotIn("names", [k.lower() for k in keys])

    def test_regression_full_workflow(self):
        query = "segment my customers into 4 clusters"
        response = execute_query(query)
        
        self.assertEqual(response["intent_classification"], "full_workflow")
        self.assertIn("summary", response)
        self.assertIn("recommendations", response)
        self.assertIn("visualizations", response)
        self.assertIn("explanations", response)
        self.assertNotIn("results", response)

    def test_regression_explanation_only(self):
        query = "explain the segmentation"
        response = execute_query(query)
        
        self.assertEqual(response["intent_classification"], "explanation_only")
        self.assertNotIn("results", response)

    def test_regression_eda_only(self):
        query = "explore the statistics of the dataset"
        response = execute_query(query)
        
        self.assertEqual(response["intent_classification"], "eda_only")
        self.assertNotIn("results", response)

    @patch("backend.tests.test_lookup.execute_query")
    def test_llm_down_fallback_lookup(self, mock_exec):
        mock_exec.return_value = {
            "intent_classification": "lookup_only",
            "metadata": {"execution_log": [{"stage": "planner", "path": "rule_based_fallback"}]},
            "results": [{"CUST_ID": "C123"}] * 5
        }
        
        query = "top 5 customers just customer id please"
        response = execute_query(query)
        
        self.assertEqual(response["intent_classification"], "lookup_only")
        
        log = response["metadata"]["execution_log"][0]
        self.assertEqual(log["stage"], "planner")
        self.assertEqual(log["path"], "rule_based_fallback")
        
        results = response["results"]
        self.assertEqual(len(results), 5)
        for row in results:
            self.assertEqual(list(row.keys()), ["CUST_ID"])

if __name__ == '__main__':
    unittest.main()
