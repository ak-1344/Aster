"""ASTER dataset manager module."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO

import pandas as pd

logger = logging.getLogger(__name__)

def process_upload(file_obj: IO[bytes], filename: str) -> dict[str, Any]:
    """Process an uploaded CSV file."""
    
    try:
        df = pd.read_csv(file_obj)
    except Exception as exc:
        logger.warning("Failed to parse uploaded CSV: %s", exc)
        raise ValueError("Invalid CSV file format or corrupted file.") from exc

    if df.empty:
        raise ValueError("Uploaded CSV is empty (no data rows).")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        raise ValueError("CSV must contain at least one numeric column for feature engineering.")

    return {
        "status": "success",
        "rows_ingested": len(df),
        "features_generated": 0,
        "dataset_id": "temp_id",
        "preview": df.head(5).to_dict(orient="records"),
    }
