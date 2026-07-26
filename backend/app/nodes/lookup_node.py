"""ASTER lookup node."""

from typing import Any
import pandas as pd

def perform_lookup(dataset_path: str, query_context: dict[str, Any]) -> dict[str, Any]:
    """Execute a simple data lookup projection and sort without analytical modeling."""
    
    requested_fields = query_context.get("requested_fields")
    top_n = query_context.get("top_n")
    unsupported_output_fields = query_context.get("unsupported_output_fields", [])
    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return {"results": [], "error": f"Failed to read dataset: {e}"}
    
    valid_fields = list(df.columns)
    if requested_fields:
        valid_fields = [f for f in requested_fields if f in df.columns]
        if not valid_fields:
            valid_fields = list(df.columns)

    if top_n is not None:
        numeric_cols = df[valid_fields].select_dtypes(include='number').columns
        if not numeric_cols.empty:
            df = df.sort_values(by=numeric_cols[0], ascending=False)
        df = df.head(top_n)

    df = df[valid_fields]
    
    return {
        "results": df.to_dict(orient="records"),
        "unsupported_output_fields": unsupported_output_fields,
        "llm_assistance": {
            "path": "deterministic",
            "reason": "Lookup performed deterministic projection."
        }
    }
