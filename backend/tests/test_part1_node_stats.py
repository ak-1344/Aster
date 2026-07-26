import unittest
import json
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.api.main import app
from backend.app.execution_graph.execution_graph import ExecutionGraph, ExecutionNode
import backend.app.scheduler.scheduler as sched
import backend.app.api.dashboard as db_api
from backend.app.decision_memory.decision_memory import _get_connection, _ensure_schema

client = TestClient(app)

class TestNodeStats(unittest.TestCase):

    def test_node_timing_persistence(self):
        graph = ExecutionGraph(
            workflow_name="test",
            intent="test",
            intent_classification="test",
            nodes=[ExecutionNode(node_name="analytics", purpose="test")]
        )
        
        original_execute_node = sched._execute_node
        
        def mock_execute_node(name, ctx):
            import time
            time.sleep(0.01)
            return {"result": "ok"}
            
        sched._execute_node = mock_execute_node
        
        try:
            outputs = sched.execute_graph(graph, {})
            self.assertIn("analytics", outputs)
            self.assertIn("_duration_ms", outputs["analytics"])
            self.assertGreaterEqual(outputs["analytics"]["_duration_ms"], 10)
            self.assertEqual(outputs["analytics"]["_status"], "success")
        finally:
            sched._execute_node = original_execute_node

    def test_dashboard_node_stats(self):
        db_file = Path("/tmp/test_decision_memory.db")
        if db_file.exists():
            db_file.unlink()
            
        original_db_path = db_api._DEFAULT_DB_PATH
        db_api._DEFAULT_DB_PATH = db_file
        
        try:
            conn = _get_connection(db_file)
            _ensure_schema(conn)
            
            summary1 = {
                "analytics": {"type": "dict", "duration_ms": 100, "status": "success"},
                "eda": {"type": "dict", "duration_ms": 200, "status": "success"}
            }
            summary2 = {
                "analytics": {"type": "dict", "duration_ms": 150, "status": "success"},
                "eda": {"type": "error", "duration_ms": 50, "status": "failed"}
            }
            
            conn.execute(
                "INSERT INTO decision_memory (query_text, query_text_normalized, node_outputs_summary, response_json, created_at) VALUES (?, ?, ?, ?, ?)",
                ("q1", "q1", json.dumps(summary1), "{}", "2023-01-01")
            )
            conn.execute(
                "INSERT INTO decision_memory (query_text, query_text_normalized, node_outputs_summary, response_json, created_at) VALUES (?, ?, ?, ?, ?)",
                ("q2", "q2", json.dumps(summary2), "{}", "2023-01-02")
            )
            conn.commit()
            
            response = client.get("/dashboard/node-stats")
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn("analytics", data)
            self.assertEqual(data["analytics"]["call_count"], 2)
            self.assertEqual(data["analytics"]["success_count"], 2)
            self.assertEqual(data["analytics"]["avg_duration_ms"], 125.0)
            
            self.assertIn("eda", data)
            self.assertEqual(data["eda"]["call_count"], 2)
            self.assertEqual(data["eda"]["success_count"], 1)
            self.assertEqual(data["eda"]["failure_count"], 1)
            self.assertEqual(data["eda"]["avg_duration_ms"], 125.0)

            self.assertIn("segmentation", data)
            self.assertEqual(data["segmentation"]["call_count"], 0)
        finally:
            db_api._DEFAULT_DB_PATH = original_db_path
            if db_file.exists():
                db_file.unlink()

if __name__ == "__main__":
    unittest.main()
