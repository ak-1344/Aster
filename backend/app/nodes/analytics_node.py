"""ASTER analytics node module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.utils.loader import DEFAULT_DATASET_PATH, load_dataset


def build_descriptive_statistics(dataset_path: str | Path | None = None) -> dict[str, Any]:
    """Return descriptive statistics for the selected dataset."""

    resolved_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    dataframe = load_dataset(resolved_path)
    numeric_frame = dataframe.select_dtypes(include="number")
    summary = {
        "dataset_path": str(resolved_path),
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "columns": list(dataframe.columns),
        "numeric_summary": {},
    }

    if not numeric_frame.empty:
        describe = numeric_frame.describe().transpose()
        for column in describe.index:
            stats = describe.loc[column]
            summary["numeric_summary"][column] = {
                "mean": round(float(stats["mean"]), 4),
                "std": round(float(stats["std"]), 4),
                "min": round(float(stats["min"]), 4),
                "max": round(float(stats["max"]), 4),
            }

    return summary