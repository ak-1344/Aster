import unittest
from fastapi.testclient import TestClient
import json

from backend.app.api.main import app
import backend.app.scheduler.scheduler as sched
from backend.app.execution_graph.execution_graph import ExecutionGraph, ExecutionNode
import threading
import time

client = TestClient(app)

class TestWebSocketPubSub(unittest.TestCase):
    def test_live_updates_sequence(self):
        events = []
        
        # We need to trigger the query in a separate thread so we can listen to the websocket
        def run_query():
            time.sleep(0.1) # wait for websocket to connect
            # We will mock the scheduler execution to avoid full pipeline
            graph = ExecutionGraph(
                workflow_name="ws_test",
                intent="test",
                intent_classification="test",
                nodes=[ExecutionNode(node_name="analytics", purpose="test")]
            )
            original_execute_node = sched._execute_node
            def mock_execute_node(name, ctx):
                time.sleep(0.05)
                return {"result": "ok"}
            sched._execute_node = mock_execute_node
            
            try:
                sched.execute_graph(graph, {})
            finally:
                sched._execute_node = original_execute_node
                
        thread = threading.Thread(target=run_query)
        thread.start()
        
        with client.websocket_connect("/dashboard/live") as websocket:
            # query_started
            data = websocket.receive_json()
            events.append(data["type"])
            self.assertEqual(data["type"], "query_started")
            
            # node_started
            data = websocket.receive_json()
            events.append(data["type"])
            self.assertEqual(data["type"], "node_started")
            self.assertEqual(data["node_name"], "analytics")
            
            # node_completed
            data = websocket.receive_json()
            events.append(data["type"])
            self.assertEqual(data["type"], "node_completed")
            self.assertEqual(data["node_name"], "analytics")
            
            # query_completed
            data = websocket.receive_json()
            events.append(data["type"])
            self.assertEqual(data["type"], "query_completed")
            self.assertEqual(data["status"], "success")
            
        thread.join()
        
        self.assertEqual(events, ["query_started", "node_started", "node_completed", "query_completed"])

    def test_graceful_disconnect(self):
        # We start a broadcast while a websocket client disconnects mid-way
        def run_broadcasts():
            time.sleep(0.1)
            from backend.app.dashboard.event_bus import broadcast
            broadcast({"type": "msg1"})
            time.sleep(0.2)
            # The client will have disconnected by now
            # This should not crash the server (which is running the event loop in test client)
            try:
                broadcast({"type": "msg2"})
            except Exception as e:
                self.fail(f"Broadcast crashed after disconnect: {e}")
                
        thread = threading.Thread(target=run_broadcasts)
        thread.start()
        
        with client.websocket_connect("/dashboard/live") as websocket:
            data = websocket.receive_json()
            self.assertEqual(data["type"], "msg1")
            # Exiting the block disconnects the websocket
            
        thread.join()

if __name__ == "__main__":
    unittest.main()
