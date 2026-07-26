"""ASTER decision engine module — SHAP/LIME surrogate explainability with rule-based fallback."""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature human-readable labels
# ---------------------------------------------------------------------------

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

# Minimum cluster members required to fit a meaningful surrogate model.
_MIN_SURROGATE_SAMPLES = 5

# Maximum seconds allowed for surrogate fitting + explanation generation.
_EXPLAINABILITY_TIMEOUT_SECONDS = 8

# Number of top feature contributions to return per explanation.
_TOP_N_FEATURES = 5

# Ratio of own-centroid distance to second-nearest-centroid distance above which
# a customer is considered borderline between segments.
_BOUNDARY_RATIO_THRESHOLD = 0.85


def _label_feature(feature_name: str) -> str:
	return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))


def _boundary_metrics_by_customer(segmentation: dict[str, Any]) -> dict[str, dict[str, Any]]:
	"""Index per-customer boundary metrics produced by the segmentation node."""

	return {
		str(metric["customer_id"]): metric
		for metric in segmentation.get("customer_boundary_metrics", [])
	}


def _append_boundary_fields(
	explanation_entry: dict[str, Any],
	boundary_metric: dict[str, Any] | None,
) -> dict[str, Any]:
	"""Attach boundary metrics and a borderline sentence when the ratio exceeds threshold."""

	if boundary_metric is None:
		return explanation_entry

	explanation_entry["nearest_alternate_cluster"] = boundary_metric["nearest_alternate_cluster"]
	explanation_entry["boundary_distance_ratio"] = boundary_metric["boundary_distance_ratio"]

	ratio = float(boundary_metric["boundary_distance_ratio"])
	if ratio > _BOUNDARY_RATIO_THRESHOLD:
		own_segment = boundary_metric["cluster_label"]
		alternate_segment = boundary_metric["nearest_alternate_cluster"]
		explanation_entry["explanation"] = (
			f"{explanation_entry['explanation']} "
			f"This customer is borderline between Segment {own_segment} and Segment {alternate_segment}."
		)

	return explanation_entry


def get_explainability_mode() -> str:
	"""Return the active explainability mode from the environment."""
	mode = os.environ.get("EXPLAINABILITY_MODE", "shap").strip().lower()
	if mode not in ("shap", "lime", "rule_based"):
		logger.warning("Invalid EXPLAINABILITY_MODE '%s', defaulting to 'shap'", mode)
		return "shap"
	return mode


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

def _fit_surrogate(features_matrix: np.ndarray, cluster_labels: list[int]) -> Any:
	"""Fit a lightweight RandomForestClassifier as a surrogate for cluster membership.

	The surrogate predicts cluster labels from the same feature set used in
	clustering. SHAP/LIME then explain predictions of this surrogate, providing
	grounded feature-importance explanations for unsupervised clustering.

	Raises:
		ValueError: When the data has fewer than _MIN_SURROGATE_SAMPLES per class
		            or only one unique label.
		ImportError: When scikit-learn is unavailable.
	"""
	from sklearn.ensemble import RandomForestClassifier

	unique_labels = set(cluster_labels)
	# Exclude DBSCAN/HDBSCAN noise label (-1) for surrogate fitting.
	non_noise_labels = {label for label in unique_labels if label != -1}
	if len(non_noise_labels) < 2:
		raise ValueError(
			f"Surrogate requires at least 2 non-noise clusters, got {len(non_noise_labels)}"
		)

	# Check minimum samples per non-noise cluster.
	label_array = np.array(cluster_labels)
	for label in non_noise_labels:
		count = int(np.sum(label_array == label))
		if count < _MIN_SURROGATE_SAMPLES:
			raise ValueError(
				f"Cluster {label} has only {count} members "
				f"(minimum {_MIN_SURROGATE_SAMPLES} required for surrogate)"
			)

	# Filter out noise-labeled rows for training.
	mask = label_array != -1
	train_features = features_matrix[mask]
	train_labels = label_array[mask]

	surrogate = RandomForestClassifier(
		n_estimators=50,
		max_depth=8,
		random_state=42,
		n_jobs=1,
	)
	surrogate.fit(train_features, train_labels)
	return surrogate


# ---------------------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------------------

def explain_with_shap(
	features_df: pd.DataFrame,
	cluster_labels: list[int],
	feature_columns: list[str],
	target_customer_index: int | None = None,
	top_n: int = _TOP_N_FEATURES,
) -> dict[str, Any]:
	"""Generate SHAP-based explanations using a surrogate model.

	Args:
		features_df: DataFrame with the feature matrix used for clustering.
		cluster_labels: Cluster label for each row in features_df.
		feature_columns: Column names of the clustering features.
		target_customer_index: If set, explain only this row. If None, explain all.
		top_n: Number of top feature contributions to return per customer.

	Returns:
		Dict with 'customer_contributions' (list of per-customer dicts) and
		'segment_aggregates' (dict of per-segment aggregated SHAP values).
	"""
	import shap

	matrix = features_df[feature_columns].to_numpy(dtype=float)
	surrogate = _fit_surrogate(matrix, cluster_labels)

	explainer = shap.TreeExplainer(surrogate)

	if target_customer_index is not None:
		rows_to_explain = matrix[target_customer_index : target_customer_index + 1]
		indices = [target_customer_index]
	else:
		rows_to_explain = matrix
		indices = list(range(len(matrix)))

	shap_values = explainer.shap_values(rows_to_explain)

	# shap_values format varies by SHAP version:
	# - Old: list of (n_samples, n_features) arrays, one per class.
	# - New (0.52+): ndarray (n_samples, n_features, n_classes).
	# Normalise to (n_samples, n_features, n_classes) for uniform access.
	if isinstance(shap_values, list):
		# Old format: list[class] of (n_samples, n_features) -> stack on axis 2.
		shap_values_3d = np.stack(shap_values, axis=-1)
	elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
		# New format: already (n_samples, n_features, n_classes).
		shap_values_3d = shap_values
	elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
		# Binary case: (n_samples, n_features) — expand to 3D.
		shap_values_3d = shap_values[:, :, np.newaxis]
	else:
		raise RuntimeError(f"Unexpected SHAP output shape: {type(shap_values)}")

	customer_contributions: list[dict[str, Any]] = []
	# Accumulate per-segment SHAP values for segment-level explanation.
	segment_shap_sums: dict[int, np.ndarray] = {}
	segment_counts: dict[int, int] = {}

	classes_list = list(surrogate.classes_)

	for i, row_index in enumerate(indices):
		cluster_label = int(cluster_labels[row_index])
		if cluster_label == -1:
			# Noise points get no SHAP explanation.
			continue

		# Extract SHAP values for the predicted class of this sample.
		class_index = classes_list.index(cluster_label)
		sample_shap = shap_values_3d[i, :, class_index]

		# Top-N by absolute SHAP value.
		abs_shap = np.abs(sample_shap)
		top_indices = np.argsort(abs_shap)[::-1][:top_n]

		contributions = []
		for fi in top_indices:
			contributions.append({
				"feature": feature_columns[fi],
				"feature_label": _label_feature(feature_columns[fi]),
				"shap_value": round(float(sample_shap[fi]), 6),
				"direction": "positive" if sample_shap[fi] > 0 else "negative",
			})

		customer_contributions.append({
			"row_index": row_index,
			"cluster_label": cluster_label,
			"contributions": contributions,
		})

		# Accumulate for segment-level aggregation.
		if cluster_label not in segment_shap_sums:
			segment_shap_sums[cluster_label] = np.zeros(len(feature_columns))
			segment_counts[cluster_label] = 0
		segment_shap_sums[cluster_label] += np.abs(sample_shap)
		segment_counts[cluster_label] += 1

	# Build segment-level aggregate explanations.
	segment_aggregates: dict[str, Any] = {}
	for label in sorted(segment_shap_sums):
		mean_abs_shap = segment_shap_sums[label] / max(segment_counts[label], 1)
		top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
		segment_aggregates[str(label)] = [
			{
				"feature": feature_columns[fi],
				"feature_label": _label_feature(feature_columns[fi]),
				"mean_abs_shap": round(float(mean_abs_shap[fi]), 6),
			}
			for fi in top_indices
		]

	return {
		"customer_contributions": customer_contributions,
		"segment_aggregates": segment_aggregates,
	}


# ---------------------------------------------------------------------------
# LIME explanations
# ---------------------------------------------------------------------------

def explain_with_lime(
	features_df: pd.DataFrame,
	cluster_labels: list[int],
	feature_columns: list[str],
	target_customer_index: int | None = None,
	top_n: int = _TOP_N_FEATURES,
) -> dict[str, Any]:
	"""Generate LIME-based explanations using a surrogate model.

	Returns the same shape as explain_with_shap so the frontend renders
	both identically regardless of which explainer ran.
	"""
	from lime.lime_tabular import LimeTabularExplainer

	matrix = features_df[feature_columns].to_numpy(dtype=float)
	surrogate = _fit_surrogate(matrix, cluster_labels)

	lime_explainer = LimeTabularExplainer(
		training_data=matrix,
		feature_names=feature_columns,
		class_names=[str(c) for c in sorted(surrogate.classes_)],
		mode="classification",
		random_state=42,
	)

	if target_customer_index is not None:
		indices = [target_customer_index]
	else:
		indices = list(range(len(matrix)))

	customer_contributions: list[dict[str, Any]] = []
	segment_weight_sums: dict[int, np.ndarray] = {}
	segment_counts: dict[int, int] = {}

	for row_index in indices:
		cluster_label = int(cluster_labels[row_index])
		if cluster_label == -1:
			continue

		class_index = list(surrogate.classes_).index(cluster_label)
		explanation = lime_explainer.explain_instance(
			matrix[row_index],
			surrogate.predict_proba,
			num_features=top_n,
			labels=(class_index,),
		)

		feature_weights = explanation.as_list(label=class_index)
		contributions = []
		for feature_desc, weight in feature_weights[:top_n]:
			# LIME feature descriptions may include range conditions;
			# extract the original feature name.
			matched_feature = None
			for col in feature_columns:
				if col in feature_desc:
					matched_feature = col
					break
			if matched_feature is None:
				matched_feature = feature_desc

			contributions.append({
				"feature": matched_feature,
				"feature_label": _label_feature(matched_feature),
				"weight": round(float(weight), 6),
				"direction": "positive" if weight > 0 else "negative",
			})

		customer_contributions.append({
			"row_index": row_index,
			"cluster_label": cluster_label,
			"contributions": contributions,
		})

		# Accumulate for segment-level aggregation.
		if cluster_label not in segment_weight_sums:
			segment_weight_sums[cluster_label] = np.zeros(len(feature_columns))
			segment_counts[cluster_label] = 0
		# Map back to feature indices for accumulation.
		for contrib in contributions:
			if contrib["feature"] in feature_columns:
				fi = feature_columns.index(contrib["feature"])
				segment_weight_sums[cluster_label][fi] += abs(contrib["weight"])
		segment_counts[cluster_label] += 1

	segment_aggregates: dict[str, Any] = {}
	for label in sorted(segment_weight_sums):
		mean_abs_weight = segment_weight_sums[label] / max(segment_counts[label], 1)
		top_indices = np.argsort(mean_abs_weight)[::-1][:top_n]
		segment_aggregates[str(label)] = [
			{
				"feature": feature_columns[fi],
				"feature_label": _label_feature(feature_columns[fi]),
				"mean_abs_weight": round(float(mean_abs_weight[fi]), 6),
			}
			for fi in top_indices
		]

	return {
		"customer_contributions": customer_contributions,
		"segment_aggregates": segment_aggregates,
	}


# ---------------------------------------------------------------------------
# Rule-based fallback (original Phase 9 implementation, preserved)
# ---------------------------------------------------------------------------

def _generate_rule_based_explanations(
	feature_columns: list[str],
	cluster_centers: list[list[float]],
	labels: list[int],
	customer_ids: list[str],
	features_path: str,
	boundary_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
	"""Produce rule-based natural-language explanations from feature distance to centroids."""

	features = pd.read_csv(features_path)
	matrix = features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
	centers = np.array(cluster_centers, dtype=float)

	customer_explanations: list[dict[str, Any]] = []
	segment_summaries: dict[int, list[str]] = {}

	for index, (customer_id, cluster_label) in enumerate(zip(customer_ids, labels)):
		cluster_index = int(cluster_label)
		if cluster_index < 0 or cluster_index >= len(centers):
			continue
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
		entry = _append_boundary_fields(
			{
				"customer_id": customer_id,
				"cluster_label": cluster_index,
				"explanation": explanation,
			},
			(boundary_metrics or {}).get(str(customer_id)),
		)
		customer_explanations.append(entry)
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


# ---------------------------------------------------------------------------
# Timeout support
# ---------------------------------------------------------------------------

class _ExplainabilityTimeout(Exception):
	"""Raised when SHAP/LIME explanation exceeds the time budget."""


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def generate_explanations(node_outputs: dict[str, Any]) -> dict[str, Any]:
	"""Produce explanations for segmentation output using the configured mode.

	Modes (controlled by EXPLAINABILITY_MODE env var):
	  - 'shap' (default): Surrogate-model SHAP explanations.
	  - 'lime': Surrogate-model LIME explanations.
	  - 'rule_based': Original centroid-distance explanations.

	Falls back to rule_based on: dependency failure, surrogate fit failure,
	timeout (8s ceiling), or clusters too small for a meaningful surrogate.
	"""
	empty = {
		"customer_explanations": [],
		"segment_summaries": [],
		"explainer_used": "none",
		"explainer_reason": "no segmentation output",
	}

	if "segmentation" not in node_outputs:
		return empty

	segmentation = node_outputs["segmentation"]
	feature_columns = segmentation.get("feature_columns", [])
	cluster_centers = segmentation.get("cluster_centers")
	labels = segmentation.get("labels", [])
	customer_ids = segmentation.get("customer_ids", [])

	if not feature_columns or not labels:
		return empty

	features_path = node_outputs.get("feature_engineering", {}).get("features_path")
	if not features_path or not Path(features_path).exists():
		return empty

	boundary_metrics = _boundary_metrics_by_customer(segmentation)
	mode = get_explainability_mode()

	# --- Rule-based fast path ---
	if mode == "rule_based":
		if not cluster_centers:
			return empty
		result = _generate_rule_based_explanations(
			feature_columns, cluster_centers, labels, customer_ids, features_path,
			boundary_metrics=boundary_metrics,
		)
		result["explainer_used"] = "rule_based"
		result["explainer_reason"] = "EXPLAINABILITY_MODE set to rule_based"
		return result

	# --- SHAP or LIME path (with timeout and fallback) ---
	features_df = pd.read_csv(features_path)
	features_df_numeric = features_df[feature_columns].apply(
		pd.to_numeric, errors="coerce"
	).fillna(0.0)

	explainer_func = explain_with_shap if mode == "shap" else explain_with_lime
	explainer_name = mode

	try:
		# Run the explainer with a thread-safe timeout.
		with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
			future = pool.submit(explainer_func, features_df_numeric, labels, feature_columns)
			raw = future.result(timeout=_EXPLAINABILITY_TIMEOUT_SECONDS)

		# Convert SHAP/LIME output to the standard response shape.
		customer_explanations = []
		for contrib_entry in raw.get("customer_contributions", []):
			row_idx = contrib_entry["row_index"]
			cid = customer_ids[row_idx] if row_idx < len(customer_ids) else str(row_idx)
			cluster_label = contrib_entry["cluster_label"]
			contributions = contrib_entry["contributions"]

			# Build a human-readable explanation from contributions.
			reasons = []
			for c in contributions[:2]:
				direction = "high" if c.get("direction") == "positive" else "low"
				reasons.append(f"{direction} {c['feature_label']}")

			explanation = (
				f"Assigned to Segment {cluster_label} primarily due to {reasons[0]}"
				+ (f" and {reasons[1]}." if len(reasons) > 1 else ".")
			)
			entry = _append_boundary_fields(
				{
					"customer_id": cid,
					"cluster_label": cluster_label,
					"explanation": explanation,
					"feature_contributions": contributions,
				},
				boundary_metrics.get(str(cid)),
			)
			customer_explanations.append(entry)

		segment_summaries = []
		for seg_label, seg_features in raw.get("segment_aggregates", {}).items():
			top_feature_names = [
				f["feature_label"] for f in seg_features[:3]
			]
			segment_summaries.append({
				"cluster_label": int(seg_label),
				"summary": (
					f"Segment {seg_label} is primarily defined by "
					f"{', '.join(top_feature_names)}."
				),
				"defining_features": seg_features,
			})

		return {
			"customer_explanations": customer_explanations,
			"segment_summaries": segment_summaries,
			"explainer_used": explainer_name,
			"explainer_reason": f"{explainer_name} surrogate-model explanation succeeded",
		}

	except concurrent.futures.TimeoutError:
		logger.warning("Explainability timeout (%ds)", _EXPLAINABILITY_TIMEOUT_SECONDS)
		fallback_reason = (
			f"rule_based_fallback: explainability exceeded "
			f"{_EXPLAINABILITY_TIMEOUT_SECONDS}s timeout"
		)
	except ImportError as exc:
		logger.warning("Explainability dependency unavailable: %s", exc)
		fallback_reason = f"rule_based_fallback: dependency failure — {exc}"
	except (ValueError, RuntimeError) as exc:
		logger.warning("Surrogate fit failure: %s", exc)
		fallback_reason = f"rule_based_fallback: surrogate fit failure — {exc}"
	except Exception as exc:
		logger.warning("Unexpected explainability error: %s", exc)
		fallback_reason = f"rule_based_fallback: unexpected error — {type(exc).__name__}: {exc}"

	# Fallback to rule-based.
	if cluster_centers:
		result = _generate_rule_based_explanations(
			feature_columns, cluster_centers, labels, customer_ids, features_path,
			boundary_metrics=boundary_metrics,
		)
	else:
		result = {"customer_explanations": [], "segment_summaries": []}
	result["explainer_used"] = "rule_based_fallback"
	result["explainer_reason"] = fallback_reason
	return result
