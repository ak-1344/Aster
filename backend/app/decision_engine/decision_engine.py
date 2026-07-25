"""ASTER decision engine module — rule-based explanations (scoped Phase 9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_LABELS: dict[str, str] = {
	"monthly_spend": "monthly spend",
	"avg_purchase_ticket": "average purchase ticket",
	"transaction_frequency_per_month": "transaction frequency",
	"cash_advance_ratio": "cash-advance ratio",
	"installment_purchase_share": "installment purchase share",
	"oneoff_purchase_share": "one-off purchase share",
	"credit_utilization_ratio": "credit utilization",
	"payment_to_minimum_ratio": "payment-to-minimum ratio",
	"full_payment_ratio": "full-payment ratio",
	"cash_advance_intensity": "cash-advance intensity",
	"credit_headroom": "credit headroom",
	"TENURE": "tenure",
}


def _label_feature(feature_name: str) -> str:
	return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))


def generate_explanations(node_outputs: dict[str, Any]) -> dict[str, Any]:
	"""Produce rule-based natural-language explanations for segmentation output."""

	if "segmentation" not in node_outputs:
		return {"customer_explanations": [], "segment_summaries": []}

	segmentation = node_outputs["segmentation"]
	feature_columns = segmentation.get("feature_columns", [])
	cluster_centers = segmentation.get("cluster_centers")
	labels = segmentation.get("labels", [])
	customer_ids = segmentation.get("customer_ids", [])

	if not feature_columns or not cluster_centers or not labels:
		return {"customer_explanations": [], "segment_summaries": []}

	features_path = node_outputs.get("feature_engineering", {}).get("features_path")
	if not features_path or not Path(features_path).exists():
		return {"customer_explanations": [], "segment_summaries": []}

	features = pd.read_csv(features_path)
	matrix = features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
	centers = np.array(cluster_centers, dtype=float)

	customer_explanations: list[dict[str, Any]] = []
	segment_summaries: dict[int, list[str]] = {}

	for index, (customer_id, cluster_label) in enumerate(zip(customer_ids, labels)):
		cluster_index = int(cluster_label)
		center = centers[cluster_index]
		row = matrix[index]
		deltas = np.abs(row - center)
		top_indices = np.argsort(deltas)[::-1][:2]

		reasons: list[str] = []
		for feature_index in top_indices:
			feature_name = feature_columns[feature_index]
			direction = "high" if row[feature_index] > center[feature_index] else "low"
			reasons.append(f"{direction} {_label_feature(feature_name)}")

		explanation = (
			f"Assigned to Segment {cluster_index} primarily due to {reasons[0]}"
			+ (f" and {reasons[1]}." if len(reasons) > 1 else ".")
		)
		customer_explanations.append(
			{
				"customer_id": customer_id,
				"cluster_label": cluster_index,
				"explanation": explanation,
			}
		)
		segment_summaries.setdefault(cluster_index, []).extend(reasons)

	segment_summary_rows = []
	for cluster_label in sorted(segment_summaries):
		top_reasons = segment_summaries[cluster_label][:6]
		segment_summary_rows.append(
			{
				"cluster_label": cluster_label,
				"summary": (
					f"Segment {cluster_label} customers are often characterized by "
					f"{', '.join(top_reasons[:3])}."
				),
			}
		)

	return {
		"customer_explanations": customer_explanations,
		"segment_summaries": segment_summary_rows,
	}
