from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import APIRouter
from backend.app.decision_memory.decision_memory import _DEFAULT_DB_PATH

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/node-stats")
def get_node_stats() -> dict[str, Any]:
    """Aggregate per-node execution stats across all decision_memory entries."""
    
    db_path = _DEFAULT_DB_PATH
    if not db_path.exists():
        # Empty state
        nodes = ["analytics", "eda", "feature_engineering", "segmentation", "recommendation", "evaluation", "visualization"]
        return {
            node: {"call_count": 0, "success_count": 0, "failure_count": 0, "avg_duration_ms": 0.0}
            for node in nodes
        }
        
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute("SELECT node_outputs_summary FROM decision_memory WHERE node_outputs_summary IS NOT NULL")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        # Table might not exist yet
        rows = []
    finally:
        conn.close()
        
    nodes = ["analytics", "eda", "feature_engineering", "segmentation", "recommendation", "evaluation", "visualization"]
    
    stats: dict[str, dict[str, Any]] = {
        node: {"call_count": 0, "success_count": 0, "failure_count": 0, "avg_duration_ms": 0.0, "_total_duration": 0}
        for node in nodes
    }
    
    for row in rows:
        try:
            summary = json.loads(row["node_outputs_summary"])
            for node_name, info in summary.items():
                if node_name not in stats:
                    stats[node_name] = {"call_count": 0, "success_count": 0, "failure_count": 0, "avg_duration_ms": 0.0, "_total_duration": 0}
                
                stats[node_name]["call_count"] += 1
                
                status = info.get("status", "success")
                if status == "success":
                    stats[node_name]["success_count"] += 1
                else:
                    stats[node_name]["failure_count"] += 1
                    
                duration = info.get("duration_ms", 0)
                stats[node_name]["_total_duration"] += duration
        except Exception:
            pass
            
    for node, data in stats.items():
        count = data["call_count"]
        if count > 0:
            data["avg_duration_ms"] = round(data["_total_duration"] / count, 2)
        del data["_total_duration"]
        
    return stats
