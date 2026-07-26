"""ASTER decision memory — exact-key query cache with SQLite persistence.

Scope: exact normalized-string-key caching only. No embeddings, no FAISS,
no vector store, no semantic similarity — deliberate scope decision;
semantic caching deferred, not a gap.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("backend/data/decision_memory.db")

# Thread-local storage for per-thread SQLite connections.
_local = threading.local()


def _normalize_query(query_text: str) -> str:
    """Normalise a query string to its exact-match cache key.

    Lowercased + whitespace-trimmed. This is deliberately not semantic —
    only identical (after normalisation) queries will hit the cache.
    """
    return " ".join(query_text.lower().split())


def _get_connection(db_path: Path) -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if necessary."""
    attr = f"_conn_{db_path}"
    conn = getattr(_local, attr, None)
    if conn is None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        setattr(_local, attr, conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the decision_memory table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_memory (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text               TEXT    NOT NULL,
            query_text_normalized    TEXT    NOT NULL,
            execution_graph_summary  TEXT,
            chosen_algorithm         TEXT,
            node_outputs_summary     TEXT,
            explanation_summary      TEXT,
            response_json            TEXT    NOT NULL,
            created_at               TEXT    NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dm_normalized
        ON decision_memory (query_text_normalized)
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db(db_path: Path | None = None) -> None:
    """Initialise the decision memory database (idempotent)."""
    path = db_path or _DEFAULT_DB_PATH
    conn = _get_connection(path)
    _ensure_schema(conn)


def lookup(
    query_text: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    """Look up an exact-match cached response for a normalised query.

    Returns the full stored response dict on cache hit, or None on miss.
    """
    path = db_path or _DEFAULT_DB_PATH
    normalized = _normalize_query(query_text)
    try:
        conn = _get_connection(path)
        _ensure_schema(conn)
        cursor = conn.execute(
            """
            SELECT response_json, created_at
            FROM decision_memory
            WHERE query_text_normalized = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (normalized,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        response = json.loads(row["response_json"])
        response["cache_hit"] = True
        response["cached_created_at"] = row["created_at"]
        return response
    except Exception:
        logger.exception("Decision memory lookup failed")
        return None


def store(
    query_text: str,
    response: dict[str, Any],
    execution_graph_summary: list[str] | None = None,
    chosen_algorithm: str | None = None,
    node_outputs_summary: dict[str, Any] | None = None,
    explanation_summary: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> None:
    """Persist a completed query response for future exact-key cache hits.

    Fire-and-forget: a write failure is logged but never blocks or fails
    the response pipeline.
    """
    path = db_path or _DEFAULT_DB_PATH
    normalized = _normalize_query(query_text)
    now = datetime.now(timezone.utc).isoformat()

    try:
        conn = _get_connection(path)
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO decision_memory (
                query_text,
                query_text_normalized,
                execution_graph_summary,
                chosen_algorithm,
                node_outputs_summary,
                explanation_summary,
                response_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_text,
                normalized,
                json.dumps(execution_graph_summary) if execution_graph_summary else None,
                chosen_algorithm,
                json.dumps(node_outputs_summary) if node_outputs_summary else None,
                json.dumps(explanation_summary) if explanation_summary else None,
                json.dumps(response, default=str),
                now,
            ),
        )
        conn.commit()
    except Exception:
        logger.exception("Decision memory write failed (fire-and-forget)")


def list_recent(
    limit: int = 20,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent cached entries for future history/session features.

    This is a read-only API exposed for potential future UI integration.
    """
    path = db_path or _DEFAULT_DB_PATH
    try:
        conn = _get_connection(path)
        _ensure_schema(conn)
        cursor = conn.execute(
            """
            SELECT id, query_text, query_text_normalized,
                   execution_graph_summary, chosen_algorithm,
                   explanation_summary, response_json,
                   created_at
            FROM decision_memory
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        logger.exception("Decision memory list_recent failed")
        return []
