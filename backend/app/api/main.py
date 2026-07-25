"""ASTER FastAPI application entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend.app.query_manager.query_manager import execute_query
from backend.app.nodes.analytics_node import build_descriptive_statistics
from backend.app.nodes.eda_node import build_exploratory_summary
from backend.app.nodes.evaluation_node import evaluate_segmentation
from backend.app.nodes.feature_engineering_node import generate_features
from backend.app.nodes.recommendation_node import build_recommendations
from backend.app.nodes.segmentation_node import segment_customers
from backend.app.nodes.visualization_node import build_visualization_payload


app = FastAPI(title="ASTER")


class QueryRequest(BaseModel):
	query: str = Field(..., min_length=1, description="Natural-language analytical query")


@app.post("/query")
def post_query(body: QueryRequest) -> dict[str, Any]:
	"""Run the canonical pipeline for a natural-language query."""

	return execute_query(body.query)


def _frontend_html() -> str:
	"""Load the temporary demo page from the frontend folder."""

	return (Path(__file__).resolve().parents[3] / "frontend" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
	"""Serve the temporary demo page."""

	return HTMLResponse(_frontend_html())


@app.post("/run-workflow")
def run_workflow() -> dict[str, Any]:
	"""FALLBACK DEMO PATH (ISSUE-002): direct node bypass for local demos.

	Canonical workflow execution is POST /query via Query Manager and Scheduler.
	Kept intentionally during hackathon time-box; do not extend this bypass.
	"""

	dataset_path = Path("backend/data/raw/CC GENERAL.csv")
	feature_path = Path("backend/data/processed/customer_features.csv")

	analytics = build_descriptive_statistics(dataset_path=dataset_path)
	exploratory = build_exploratory_summary(dataset_path=dataset_path)
	features, saved_path = generate_features(dataset_path=dataset_path, output_path=feature_path)
	segmentation = segment_customers(features=features, n_clusters=3)
	labels = segmentation["labels"]
	evaluation = evaluate_segmentation(features=features, labels=labels)
	recommendations = build_recommendations(features=features, labels=labels)
	visualizations = build_visualization_payload(features=features, labels=labels)

	cluster_table = []
	customer_recommendation_lookup = {
		item["customer_id"]: item["primary_recommendation"]
		for item in recommendations["customer_recommendations"]
	}
	for row in segmentation["clustered_customers"]:
		cluster_label = int(row["cluster_label"])
		cluster_table.append(
			{
				"customer_id": row["CUST_ID"],
				"cluster_label": cluster_label,
				"tier": recommendations["cluster_tiers"][cluster_label],
				"primary_recommendation": customer_recommendation_lookup[row["CUST_ID"]],
			}
		)

	return {
		"dataset_status": {
			"dataset_path": str(dataset_path),
			"feature_output_path": str(saved_path),
			"rows": analytics["row_count"],
			"columns": analytics["column_count"],
			"customer_identifier": exploratory["customer_identifier"],
		},
		"analytics": analytics,
		"exploratory": exploratory,
		"segmentation": segmentation,
		"evaluation": evaluation,
		"cluster_recommendations": recommendations["cluster_recommendations"],
		"customer_recommendations": recommendations["customer_recommendations"],
		"visualizations": visualizations,
		"evaluation_clusters": [
			{"cluster_label": int(key), "customer_count": int(value)}
			for key, value in evaluation["cluster_sizes"].items()
		],
		"cluster_table": cluster_table,
	}