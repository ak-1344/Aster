import unittest
import json
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app.api.main import app
import backend.app.api.dashboard as db_api
from backend.app.decision_memory.decision_memory import _get_connection, _ensure_schema

client = TestClient(app)

class TestGraphEndpoint(unittest.TestCase):
    def test_get_query_graph(self):
        db_file = Path("/tmp/test_decision_memory_graph.db")
        if db_file.exists():
            db_file.unlink()
            
        original_db_path = db_api._DEFAULT_DB_PATH
        db_api._DEFAULT_DB_PATH = db_file
        
        try:
            conn = _get_connection(db_file)
            _ensure_schema(conn)
            
            graph_summary = [
                {"node_name": "analytics", "dependencies": []},
                {"node_name": "eda", "dependencies": ["analytics"]},
                {"node_name": "skipped_node", "dependencies": ["eda"]}
            ]
            node_summary = {
                "analytics": {"type": "dict", "duration_ms": 100, "status": "success"},
                "eda": {"type": "error", "duration_ms": 50, "status": "failed"}
            }
            
            conn.execute(
                "INSERT INTO decision_memory (query_text, query_text_normalized, execution_graph_summary, node_outputs_summary, response_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("q1", "q1", json.dumps(graph_summary), json.dumps(node_summary), "{}", "2023-01-01")
            )
            conn.commit()
            
            response = client.get("/dashboard/queries/1/graph")
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0]["node_name"], "analytics")
            self.assertEqual(data[0]["status"], "success")
            self.assertEqual(data[0]["duration_ms"], 100)
            
            self.assertEqual(data[1]["node_name"], "eda")
            self.assertEqual(data[1]["status"], "failed")
            self.assertEqual(data[1]["duration_ms"], 50)
            self.assertEqual(data[1]["dependencies"], ["analytics"])
            
            self.assertEqual(data[2]["node_name"], "skipped_node")
            self.assertEqual(data[2]["status"], "skipped")
            
            response_404 = client.get("/dashboard/queries/99/graph")
            self.assertEqual(response_404.status_code, 404)
        finally:
            db_api._DEFAULT_DB_PATH = original_db_path
            if db_file.exists():
                db_file.unlink()

if __name__ == "__main__":
    unittest.main()
