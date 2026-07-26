"""Decision memory tests — cache hit, cache miss, exact-key behavior, and scheduler bypass."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.decision_memory.decision_memory import (
    _normalize_query,
    init_db,
    list_recent,
    lookup,
    store,
)


class DecisionMemoryExactKeyTests(unittest.TestCase):
    """Tests for exact-key cache behavior."""

    def setUp(self) -> None:
        """Create a temporary database for each test."""
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self._tmpdir) / "test_dm.db"
        init_db(self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary database."""
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_cache_miss_returns_none(self) -> None:
        """A query not previously stored returns None."""
        result = lookup("segment customers", db_path=self.db_path)
        self.assertIsNone(result)

    def test_store_and_lookup_returns_cache_hit(self) -> None:
        """Storing a response and looking up the same query returns cache_hit=True."""
        query = "segment customers into 3 clusters"
        response = {"workflow_name": "segmentation_workflow", "data": "test"}

        store(query, response, db_path=self.db_path)
        result = lookup(query, db_path=self.db_path)

        self.assertIsNotNone(result)
        self.assertTrue(result["cache_hit"])
        self.assertEqual(result["workflow_name"], "segmentation_workflow")
        self.assertIn("cached_created_at", result)

    def test_second_identical_query_is_cache_hit(self) -> None:
        """Running the same query twice — second call returns cache_hit."""
        query = "segment customers into 3 clusters"
        response = {"workflow_name": "segmentation_workflow"}

        store(query, response, db_path=self.db_path)

        first_lookup = lookup(query, db_path=self.db_path)
        second_lookup = lookup(query, db_path=self.db_path)

        self.assertIsNotNone(first_lookup)
        self.assertIsNotNone(second_lookup)
        self.assertTrue(first_lookup["cache_hit"])
        self.assertTrue(second_lookup["cache_hit"])

    def test_different_phrasing_same_intent_is_cache_miss(self) -> None:
        """Two differently-phrased queries must both be cache misses.

        This is intentional exact-key behavior, not a gap — semantic
        caching is explicitly deferred.
        """
        query_a = "segment customers into 3 clusters"
        query_b = "please group my customers into three segments"

        response_a = {"workflow_name": "segmentation_workflow"}
        store(query_a, response_a, db_path=self.db_path)

        # query_b should be a cache miss despite same intent.
        result_b = lookup(query_b, db_path=self.db_path)
        self.assertIsNone(result_b, "Different phrasing must be a cache miss (exact-key only)")

    def test_normalization_handles_whitespace_and_case(self) -> None:
        """Cache key normalises case and extra whitespace."""
        query_original = "  Segment  Customers   into 3 Clusters  "
        query_normalized = "segment customers into 3 clusters"

        response = {"workflow_name": "test"}
        store(query_original, response, db_path=self.db_path)

        # Same content with different casing/whitespace should hit.
        result = lookup(query_normalized, db_path=self.db_path)
        self.assertIsNotNone(result)
        self.assertTrue(result["cache_hit"])

    def test_list_recent_returns_stored_entries(self) -> None:
        """list_recent() returns previously stored entries for future UI."""
        store("query 1", {"data": 1}, db_path=self.db_path)
        store("query 2", {"data": 2}, db_path=self.db_path)

        recent = list_recent(limit=10, db_path=self.db_path)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["query_text"], "query 2")  # Most recent first.

    def test_normalize_query_function(self) -> None:
        """_normalize_query lowercases and trims whitespace."""
        self.assertEqual(_normalize_query("  Hello   World  "), "hello world")
        self.assertEqual(_normalize_query("SEGMENT"), "segment")


class DecisionMemoryCacheBypassTests(unittest.TestCase):
    """Test that cache hit correctly bypasses the scheduler."""

    def test_cache_hit_skips_scheduler(self) -> None:
        """On cache hit, execute_query must NOT invoke the scheduler.

        Asserts via mock/spy that execute_graph is never called when
        the query is already cached.
        """
        from backend.app.query_manager.query_manager import execute_query

        # First call: run the full pipeline to populate the cache.
        query = "segment customers into 3 clusters"
        first_result = execute_query(query)
        self.assertIsNotNone(first_result)

        # Second call: should hit the cache and skip the scheduler.
        with patch(
            "backend.app.query_manager.query_manager.execute_graph"
        ) as mock_execute_graph:
            second_result = execute_query(query)

            # execute_graph must NOT have been called.
            mock_execute_graph.assert_not_called()

            # Result should be a cache hit.
            self.assertTrue(
                second_result.get("cache_hit"),
                "Second call should return cache_hit=True",
            )


if __name__ == "__main__":
    unittest.main()
