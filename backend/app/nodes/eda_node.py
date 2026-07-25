"""Dataset-understanding node for ASTER."""

from __future__ import annotations

from pathlib import Path

from utils.loader import DatasetUnderstandingReport, build_dataset_understanding_report


def build_report(dataset_path: str | Path | None = None) -> DatasetUnderstandingReport:
	"""Build a dataset-understanding report for the selected dataset."""

	return build_dataset_understanding_report(dataset_path)
"""ASTER EDA node module."""

pass