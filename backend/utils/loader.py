"""Dataset loading and profiling helpers for ASTER."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "CC GENERAL.csv"


@dataclass(slots=True)
class DatasetUnderstandingReport:
	"""Structured summary of the selected ASTER dataset."""

	dataset_path: Path
	row_count: int
	column_count: int
	columns: list[str]
	dtypes: dict[str, str]
	missing_values: dict[str, int]
	missing_percent: dict[str, float]
	numeric_summary: dict[str, dict[str, float]] = field(default_factory=dict)
	categorical_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
	customer_identifier: str | None = None
	transaction_identifier: str | None = None
	notes: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		"""Return a JSON-serializable representation of the report."""

		return {
			"dataset_path": str(self.dataset_path),
			"row_count": self.row_count,
			"column_count": self.column_count,
			"columns": self.columns,
			"dtypes": self.dtypes,
			"missing_values": self.missing_values,
			"missing_percent": self.missing_percent,
			"numeric_summary": self.numeric_summary,
			"categorical_summary": self.categorical_summary,
			"customer_identifier": self.customer_identifier,
			"transaction_identifier": self.transaction_identifier,
			"notes": self.notes,
		}

	def to_markdown(self) -> str:
		"""Render the report as a compact Markdown document."""

		lines = [
			"# Dataset Understanding Report",
			"",
			f"- Dataset: `{self.dataset_path.name}`",
			f"- Rows: `{self.row_count}`",
			f"- Columns: `{self.column_count}`",
			f"- Customer identifier: `{self.customer_identifier or 'Not identified'}`",
			f"- Transaction identifier: `{self.transaction_identifier or 'Not identified'}`",
			"",
			"## Columns",
		]

		for column in self.columns:
			lines.append(f"- `{column}` ({self.dtypes[column]})")

		lines.extend(["", "## Missing Values"])
		for column, count in self.missing_values.items():
			lines.append(f"- `{column}`: {count} ({self.missing_percent[column]:.2f}%)")

		if self.numeric_summary:
			lines.extend(["", "## Numeric Summary"])
			for column, stats in self.numeric_summary.items():
				summary_bits = ", ".join(
					f"{metric}={value:.4f}" for metric, value in stats.items() if pd.notna(value)
				)
				lines.append(f"- `{column}`: {summary_bits}")

		if self.notes:
			lines.extend(["", "## Notes"])
			lines.extend(f"- {note}" for note in self.notes)

		return "\n".join(lines)


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
	"""Load the ASTER dataset from disk."""

	resolved_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
	return pd.read_csv(resolved_path)


def identify_identifier_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
	"""Identify customer and transaction identifier columns when present."""

	customer_identifier = None
	transaction_identifier = None

	for column in df.columns:
		normalized = "".join(character for character in column.lower() if character.isalnum())
		unique_ratio = df[column].nunique(dropna=False) / max(len(df), 1)

		if customer_identifier is None and (
			"custid" in normalized
			or ("customer" in normalized and "id" in normalized)
			or (normalized.endswith("id") and unique_ratio > 0.9)
		):
			customer_identifier = column
			continue

		if transaction_identifier is None and (
			"transactionid" in normalized
			or ("trans" in normalized and "id" in normalized)
			or ("txn" in normalized and "id" in normalized)
		):
			transaction_identifier = column

	return customer_identifier, transaction_identifier


def build_dataset_understanding_report(dataset_path: str | Path | None = None) -> DatasetUnderstandingReport:
	"""Build a reusable dataset-understanding report for the selected dataset."""

	resolved_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
	dataframe = load_dataset(resolved_path)

	missing_values = dataframe.isna().sum().astype(int).to_dict()
	missing_percent = {
		column: round((count / len(dataframe)) * 100, 4) if len(dataframe) else 0.0
		for column, count in missing_values.items()
	}

	numeric_frame = dataframe.select_dtypes(include="number")
	numeric_summary: dict[str, dict[str, float]] = {}
	if not numeric_frame.empty:
		numeric_describe = numeric_frame.describe().transpose()
		for column, row in numeric_describe.iterrows():
			numeric_summary[column] = {
				metric: float(value)
				for metric, value in row.items()
				if pd.notna(value)
			}

	categorical_summary: dict[str, dict[str, Any]] = {}
	categorical_frame = dataframe.select_dtypes(exclude="number")
	for column in categorical_frame.columns:
		series = categorical_frame[column]
		categorical_summary[column] = {
			"unique": int(series.nunique(dropna=False)),
			"top": series.mode(dropna=False).iloc[0] if not series.mode(dropna=False).empty else None,
			"frequency": int(series.value_counts(dropna=False).iloc[0]) if not series.empty else 0,
		}

	customer_identifier, transaction_identifier = identify_identifier_columns(dataframe)
	notes = ["Dataset is customer-level, so no transaction identifier was detected."]
	if transaction_identifier is not None:
		notes = ["A transaction-style identifier was detected alongside the customer data."]

	return DatasetUnderstandingReport(
		dataset_path=resolved_path,
		row_count=int(dataframe.shape[0]),
		column_count=int(dataframe.shape[1]),
		columns=list(dataframe.columns),
		dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
		missing_values=missing_values,
		missing_percent=missing_percent,
		numeric_summary=numeric_summary,
		categorical_summary=categorical_summary,
		customer_identifier=customer_identifier,
		transaction_identifier=transaction_identifier,
		notes=notes,
	)
