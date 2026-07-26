"""ASTER context builder module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd

from backend.utils.loader import DEFAULT_DATASET_PATH


SEGMENTATION_KEYWORDS = {
	"cluster",
	"clusters",
	"segment",
	"segmentation",
	"persona",
	"persona",
	"group",
	"grouping",
}
DESCRIPTIVE_KEYWORDS = {
	"describe",
	"descriptive",
	"summary",
	"statistics",
	"statistical",
	"eda",
	"overview",
}


@dataclass(slots=True)
class QueryContext:
	"""Structured query context for the planner."""

	raw_query: str
	normalized_query: str
	intent: str
	entities: list[str] = field(default_factory=list)
	filters: dict[str, Any] = field(default_factory=dict)
	output_format: str = "table"
	dataset_path: Path = DEFAULT_DATASET_PATH
	notes: list[str] = field(default_factory=list)
	unsupported_filters: list[dict[str, str]] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		"""Return a JSON-serializable representation of the context."""

		return {
			"raw_query": self.raw_query,
			"normalized_query": self.normalized_query,
			"intent": self.intent,
			"entities": self.entities,
			"filters": self.filters,
			"output_format": self.output_format,
			"dataset_path": str(self.dataset_path),
			"notes": self.notes,
			"unsupported_filters": self.unsupported_filters,
		}


def normalize_query(query: str) -> str:
	"""Normalize whitespace and punctuation for intent routing."""

	normalized = query.lower().strip()
	normalized = re.sub(r"[^a-z0-9\s_]", " ", normalized)
	normalized = re.sub(r"\s+", " ", normalized)
	return normalized.strip()


def infer_intent(normalized_query: str) -> str:
	"""Infer the request intent from the normalized query text."""

	tokens = set(normalized_query.split())
	if tokens & SEGMENTATION_KEYWORDS:
		return "segmentation"
	if tokens & DESCRIPTIVE_KEYWORDS:
		return "descriptive"
	return "descriptive"


def extract_entities(normalized_query: str) -> list[str]:
	"""Extract simple keyword entities from the query."""

	entities: list[str] = []
	if any(keyword in normalized_query for keyword in SEGMENTATION_KEYWORDS):
		entities.append("customer_clusters")
	if any(keyword in normalized_query for keyword in DESCRIPTIVE_KEYWORDS):
		entities.append("dataset_summary")
	if "recommend" in normalized_query:
		entities.append("recommendations")
	if "visual" in normalized_query or "chart" in normalized_query or "plot" in normalized_query:
		entities.append("visualization")
	return entities


def extract_and_validate_filters(normalized_query: str, dataset_path: Path) -> list[dict[str, str]]:
	"""Extract requested filters and validate against dataset columns."""
	
	FILTER_DOMAINS = {
		"age": ["age", "aged", "old", "young", "years"],
		"city/location": ["city", "location", "chennai", "mumbai", "delhi", "bangalore", "country", "state", "region", "zip", "area"],
		"gender": ["gender", "male", "female", "men", "women", "sex"],
		"account balance": ["account", "balance", "money", "funds"],
		"purchase frequency": ["purchase", "frequency", "buy", "often"],
		"credit limit": ["credit", "limit"],
		"tenure": ["tenure", "duration", "time", "months"],
		"payments": ["payment", "payments", "paid"],
	}
	
	requested_domains = []
	tokens = set(normalized_query.split())
	for domain, keywords in FILTER_DOMAINS.items():
		if any(keyword in tokens for keyword in keywords):
			requested_domains.append(domain)
			
	if not requested_domains:
		return []
		
	try:
		df = pd.read_csv(dataset_path, nrows=0)
		columns = [col.lower() for col in df.columns]
	except Exception:
		columns = []
		
	unsupported_filters = []
	for domain in requested_domains:
		domain_words = set(re.findall(r'[a-z]+', domain.lower()))
		
		is_supported = False
		for col in columns:
			if any(word in col for word in domain_words if len(word) > 2):
				is_supported = True
				break
				
		if not is_supported:
			unsupported_filters.append({
				"requested": domain,
				"reason": "no matching column in current dataset"
			})
			
	return unsupported_filters


def build_context(query: str, dataset_path: str | Path | None = None) -> dict[str, Any]:
	"""Build a structured planner context from a user query."""

	resolved_dataset_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
	normalized_query = normalize_query(query)
	intent = infer_intent(normalized_query)
	output_format = "chart" if any(keyword in normalized_query for keyword in {"chart", "plot", "visual", "graph"}) else "table"

	filters: dict[str, Any] = {}
	cluster_match = re.search(r"(\d+)\s*(?:clusters?|segments?)", normalized_query)
	if not cluster_match:
		cluster_match = re.search(r"(?:cluster|segment)\b.*?(\d+)", normalized_query)
	if cluster_match:
		filters["n_clusters"] = int(cluster_match.group(1))

	context = QueryContext(
		raw_query=query,
		normalized_query=normalized_query,
		intent=intent,
		entities=extract_entities(normalized_query),
		filters=filters,
		output_format=output_format,
		dataset_path=resolved_dataset_path,
		unsupported_filters=extract_and_validate_filters(normalized_query, resolved_dataset_path),
	)
	return context.to_dict()