"""EDA node for ASTER."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.utils.loader import DEFAULT_DATASET_PATH, build_dataset_understanding_report, load_dataset


def build_report(dataset_path: str | Path | None = None) -> Any:
    """Build a dataset-understanding report for the selected dataset."""

    return build_dataset_understanding_report(dataset_path)


def build_exploratory_summary(dataset_path: str | Path | None = None) -> dict[str, Any]:
    """Return a reusable exploratory summary for downstream analysis."""

    resolved_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    dataframe = load_dataset(resolved_path)
    report = build_dataset_understanding_report(resolved_path)

    numeric_frame = dataframe.select_dtypes(include="number")
    numeric_correlations: dict[str, dict[str, float]] = {}
    if not numeric_frame.empty:
        numeric_correlations = numeric_frame.corr(numeric_only=True).round(4).fillna(0.0).to_dict()

    return {
        "dataset_path": str(resolved_path),
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "columns": list(dataframe.columns),
        "missing_values": report.missing_values,
        "missing_percentage": report.missing_percent,
        "customer_identifier": report.customer_identifier,
        "transaction_identifier": report.transaction_identifier,
        "notes": report.notes,
        "numeric_correlations": numeric_correlations,
        "sample_rows": dataframe.head(5).to_dict(orient="records"),
    }