import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from backend.app.context_builder.context_builder import build_context, extract_and_validate_filters
from backend.app.query_manager.query_manager import execute_query

class TestUnsupportedFilters(unittest.TestCase):
    def test_unsupported_filter_extraction(self):
        # Query with unsupported filters
        query = "users in Chennai between age 20 and 24 with less than 100 in their account"
        ctx = build_context(query)
        unsupported = ctx.get("unsupported_filters", [])
        requested = [f["requested"] for f in unsupported]
        self.assertIn("age", requested)
        self.assertIn("city/location", requested)
        self.assertNotIn("account balance", requested) # Supported!

    def test_supported_filter_extraction(self):
        # Fully supported
        query = "segment customers by balance and purchase frequency"
        ctx = build_context(query)
        unsupported = ctx.get("unsupported_filters")
        self.assertEqual(unsupported, [])
        
    def test_custom_dataset(self):
        # Mock a custom dataset
        custom_csv = Path("/tmp/custom_data.csv")
        df = pd.DataFrame({"age": [20, 25], "city": ["Chennai", "Delhi"], "balance": [100, 200]})
        df.to_csv(custom_csv, index=False)
        
        query = "users in Chennai between age 20 and 24 with less than 100 in their account"
        ctx = build_context(query, dataset_path=custom_csv)
        unsupported = ctx.get("unsupported_filters", [])
        requested = [f["requested"] for f in unsupported]
        self.assertEqual(requested, []) # All supported now!
        
        custom_csv.unlink()

if __name__ == '__main__':
    unittest.main()
