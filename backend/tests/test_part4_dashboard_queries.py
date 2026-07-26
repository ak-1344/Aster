import pytest
from fastapi.testclient import TestClient
from backend.app.api.main import app
from backend.app.decision_memory.decision_memory import _DEFAULT_DB_PATH, init_db, store
import sqlite3
import json

import backend.app.api.dashboard as db_api
from pathlib import Path

@pytest.fixture(autouse=True)
def setup_teardown_db():
    db_file = Path("/tmp/test_dashboard_queries.db")
    if db_file.exists():
        db_file.unlink()
    
    original_db_path = db_api._DEFAULT_DB_PATH
    db_api._DEFAULT_DB_PATH = db_file
    
    # We must also patch decision_memory module's _DEFAULT_DB_PATH since list_recent uses it without args in dashboard.py, wait, list_recent uses path = db_path or _DEFAULT_DB_PATH, so if we don't mock decision_memory._DEFAULT_DB_PATH, it uses the original.
    import backend.app.decision_memory.decision_memory as dm
    original_dm_db_path = dm._DEFAULT_DB_PATH
    dm._DEFAULT_DB_PATH = db_file
    
    init_db(db_file)
    
    yield db_file
    
    db_api._DEFAULT_DB_PATH = original_db_path
    dm._DEFAULT_DB_PATH = original_dm_db_path
    
    if db_file.exists():
        db_file.unlink()

def test_dashboard_queries_endpoint(setup_teardown_db):
    db_file = setup_teardown_db
    client = TestClient(app)
    
    # Store some fake entries in decision memory
    store(
        query_text="query 1",
        response={"cache_hit": False},
        execution_graph_summary=["node1"],
        chosen_algorithm="kmeans",
        node_outputs_summary={"node1": {"status": "success", "duration_ms": 100}},
        explanation_summary={"explainer_used": "shap"},
        db_path=db_file
    )
    store(
        query_text="query 2",
        response={"cache_hit": True},
        execution_graph_summary=["node1", "node2"],
        chosen_algorithm="dbscan",
        node_outputs_summary={"node1": {"status": "success"}, "node2": {"status": "success"}},
        explanation_summary={"explainer_used": "lime"},
        db_path=db_file
    )
    store(
        query_text="query 3",
        response={"cache_hit": False},
        execution_graph_summary=[],
        chosen_algorithm=None,
        node_outputs_summary={},
        explanation_summary={"explainer_used": "none"},
        db_path=db_file
    )
    
    # Need a tiny sleep if timestamps were identical, but SQLite created_at should be fine, actually wait, they might have same timestamp if run very fast.
    # The API orders by created_at DESC, so it might be non-deterministic if they are exactly the same.
    # But let's just assert the length and presence of fields.
    
    response = client.get("/dashboard/queries")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    
    # Verify the structure of a returned row
    # The first one returned should be "query 3" assuming chronological insertion order.
    # We can just check that all required fields are present
    for row in data:
        assert "id" in row
        assert "query_text" in row
        assert "chosen_algorithm" in row
        assert "cache_hit" in row
        assert "explainer_used" in row
        assert "created_at" in row
        
    query_texts = [row["query_text"] for row in data]
    assert "query 1" in query_texts
    assert "query 2" in query_texts
    assert "query 3" in query_texts
