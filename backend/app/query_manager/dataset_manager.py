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

    # Create timestamped filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dataset_id = f"upload_{timestamp}"
    raw_path = Path("backend/data/raw") / f"{dataset_id}.csv"
    processed_path = Path("backend/data/processed") / f"{dataset_id}.csv"

    # Ensure directories exist
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw file
    df.to_csv(raw_path, index=False)
    
    # Run feature engineering synchronously
    try:
        from backend.app.nodes.feature_engineering_node import generate_features
        features_df, _ = generate_features(
            dataset_path=raw_path,
            output_path=processed_path
        )
    except Exception as exc:
        # Clean up raw file on pipeline failure
        if raw_path.exists():
            raw_path.unlink()
        logger.exception("Feature engineering failed during upload")
        raise RuntimeError(f"Feature engineering pipeline failed: {exc}") from exc

    return {
        "status": "success",
        "rows_ingested": len(df),
        "features_generated": len(features_df.columns),
        "dataset_id": dataset_id,
        "preview": df.head(5).to_dict(orient="records"),
    }
