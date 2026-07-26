"""ASTER context builder module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.utils.loader import DEFAULT_DATASET_PATH


# Minimum cosine similarity to accept a TF-IDF semantic intent match.
_SEMANTIC_SIMILARITY_THRESHOLD = 0.18

INTENT_EXAMPLES: dict[str, list[str]] = {
	"segmentation": [
		"segment customers into clusters",
		"group customers by spending behavior",
		"cluster customers into personas",
		"find customers suitable for investment products",
		"which customers are becoming inactive",
		"target customers for retention campaigns",
		"identify customers at risk of churn",
		"recommend products for high value customers",
		"cross sell premium offers to eligible customers",
		"find customers for loan upgrade campaigns",
		"marketing segmentation for customer targeting",
		"convert dormant customers with win back offers",
		"who should i market our high end credit card to",
		"market premium credit cards to the right customers",
		"which people are relying heavily on cash advances",
		"find customers with heavy cash advance usage",
		"target customers for high end credit card marketing",
	],
	"descriptive": [
		"show descriptive statistics for the dataset",
		"summarize the dataset overview",
		"what are the summary stats",
		"explore missing values in the dataset",
		"show distributions of numeric columns",
		"correlation analysis of customer features",
		"profile the dataset with eda",
		"give me an overview of the data",
		"statistical summary of all columns",
	],
	"explanation": [
		"explain why this customer is in this segment",
		"why was customer assigned to cluster",
		"explain the segment assignment for a customer",
		"how do you interpret this cluster membership",
		"explain customer lookup reasons",
		"clarify the risk explanation for this segment",
		"what does this segment mean for the customer",
		"why was customer placed in this segment",
		"is this customer suspicious",
		"look up why this customer was flagged",
	],
}

_CUST_ID_PATTERN = re.compile(r"\bC\d{4,}\b", re.IGNORECASE)

# DEPRECATED: legacy exact-token keyword lists retained for reference and gradual
# migration. Primary intent routing now uses TF-IDF semantic similarity via
# classify_intent_semantically(). Do not add new keywords here.
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
BUSINESS_INTENT_KEYWORDS = {
	"recommend",
	"recommendation",
	"recommendations",
	"suitable",
	"eligible",
	"target",
	"targeted",
	"campaign",
	"campaigns",
	"retention",
	"inactive",
	"dormant",
	"churn",
	"upgrade",
	"priority",
	"cross",
	"upsell",
	"investment",
	"premium",
	"offer",
	"offers",
	"loan",
}
BUSINESS_INTENT_PHRASES = {
	"cross sell",
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
ALL_INTENT_KEYWORDS = SEGMENTATION_KEYWORDS | BUSINESS_INTENT_KEYWORDS | DESCRIPTIVE_KEYWORDS
RULE_ENGINE_TIER_NAMES = {"priority", "regular", "dormant"}
THRESHOLD_PHRASES = {
	"high value",
	"high spending",
	"high spender",
	"high spenders",
	"big spender",
	"big spenders",
	"low balance",
	"low spending",
	"heavy user",
	"heavy users",
	"top spenders",
	"big spend",
}

# Generic tokens ignored when deciding if a TF-IDF match reflects real intent
# (e.g. "list customers in Chennai" should not bypass clarification via "customers").
_GENERIC_INTENT_TOKENS = {
	"a",
	"all",
	"an",
	"and",
	"are",
	"customer",
	"customers",
	"find",
	"for",
	"get",
	"give",
	"in",
	"is",
	"list",
	"me",
	"my",
	"of",
	"or",
	"show",
	"that",
	"the",
	"this",
	"to",
	"what",
	"which",
	"who",
	"with",
}


def normalize_query(query: str) -> str:
	"""Normalize whitespace and punctuation for intent routing."""

	normalized = query.lower().strip()
	normalized = re.sub(r"[^a-z0-9\s_]", " ", normalized)
	normalized = re.sub(r"\s+", " ", normalized)
	return normalized.strip()


def _build_tfidf_intent_cache() -> tuple[
	TfidfVectorizer,
	dict[str, Any],
	list[str],
	list[str],
]:
	"""Fit and cache TF-IDF vectors for labeled intent example queries."""

	all_examples: list[str] = []
	example_categories: list[str] = []
	for category, phrases in INTENT_EXAMPLES.items():
		for phrase in phrases:
			all_examples.append(normalize_query(phrase))
			example_categories.append(category)

	vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
	example_matrix = vectorizer.fit_transform(all_examples)

	category_vectors: dict[str, Any] = {}
	for category in INTENT_EXAMPLES:
		indices = [index for index, label in enumerate(example_categories) if label == category]
		centroid = example_matrix[indices].mean(axis=0)
		category_vectors[category] = np.asarray(centroid)

	return vectorizer, category_vectors, example_categories, all_examples


_TFIDF_VECTORIZER, _CATEGORY_VECTORS, _EXAMPLE_CATEGORIES, _EXAMPLE_PHRASES = (
	_build_tfidf_intent_cache()
)


def classify_intent_semantically(query: str) -> dict[str, Any]:
	"""Classify query intent via cosine similarity to canonical example phrases."""

	normalized_query = normalize_query(query)
	query_vector = _TFIDF_VECTORIZER.transform([normalized_query])

	best_category = "descriptive"
	best_score = 0.0
	best_example = ""

	for category, category_vector in _CATEGORY_VECTORS.items():
		score = float(cosine_similarity(query_vector, category_vector)[0][0])
		if score <= best_score:
			continue

		best_score = score
		best_category = category
		category_indices = [
			index for index, label in enumerate(_EXAMPLE_CATEGORIES) if label == category
		]
		category_example_matrix = _TFIDF_VECTORIZER.transform(
			[_EXAMPLE_PHRASES[index] for index in category_indices]
		)
		example_scores = cosine_similarity(query_vector, category_example_matrix)[0]
		best_local_index = int(np.argmax(example_scores))
		best_example = _EXAMPLE_PHRASES[category_indices[best_local_index]]

	return {
		"category": best_category,
		"similarity_score": round(best_score, 4),
		"matched_example": best_example,
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
	intent_routing: dict[str, Any] = field(default_factory=dict)

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
			"intent_routing": self.intent_routing,
		}


def infer_intent(normalized_query: str, semantic: dict[str, Any] | None = None) -> str:
	"""Infer the request intent from TF-IDF semantic similarity."""

	# Direct entity lookup: a CUST_ID mention routes to explanation/customer insight.
	if _CUST_ID_PATTERN.search(normalized_query or ""):
		return "explanation"

	routing = semantic or classify_intent_semantically(normalized_query)
	if routing["similarity_score"] < _SEMANTIC_SIMILARITY_THRESHOLD:
		return "descriptive"

	category = routing["category"]
	if category == "segmentation":
		return "segmentation"
	if category == "explanation":
		return "explanation"
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


def _intent_bearing_overlap(normalized_query: str, matched_example: str) -> bool:
	"""Return True when query and matched example share non-generic intent tokens."""

	query_tokens = set(normalized_query.split()) - _GENERIC_INTENT_TOKENS
	example_tokens = set(normalize_query(matched_example).split()) - _GENERIC_INTENT_TOKENS
	return bool(query_tokens & example_tokens)


def _has_intent_signal(normalized_query: str) -> bool:
	"""Return True when the query matches a known routing signal."""

	# DEPRECATED: legacy keyword fallback for ambiguity checks during migration.
	tokens = set(normalized_query.split())
	if tokens & ALL_INTENT_KEYWORDS:
		return True
	if any(phrase in normalized_query for phrase in BUSINESS_INTENT_PHRASES):
		return True

	semantic = classify_intent_semantically(normalized_query)
	if semantic["similarity_score"] < _SEMANTIC_SIMILARITY_THRESHOLD:
		return False
	return _intent_bearing_overlap(normalized_query, semantic["matched_example"])


def _has_resolvable_intent_context(query_context: dict[str, Any]) -> bool:
	"""Return True when the query has enough context to proceed without clarification."""

	normalized_query = str(query_context.get("normalized_query", ""))
	if _has_intent_signal(normalized_query):
		return True
	if query_context.get("filters"):
		return True
	# Entity hints alone (e.g. customer_clusters from generic "customers") are not enough
	# when unsupported filters block the query.
	return False


def _mentions_threshold_without_anchor(normalized_query: str) -> bool:
	"""Return True for relative value terms with no numeric anchor or known tier name."""

	tokens = set(normalized_query.split())
	if tokens & RULE_ENGINE_TIER_NAMES:
		return False
	if re.search(r"\d", normalized_query):
		return False
	return any(phrase in normalized_query for phrase in THRESHOLD_PHRASES)


def _is_query_too_vague(normalized_query: str) -> bool:
	"""Return True when the query is too short and lacks any intent keyword."""

	tokens = normalized_query.split()
	return len(tokens) < 3 and not _has_intent_signal(normalized_query)


def detect_ambiguity(
	query_context: dict[str, Any],
	dataset_path: Path | None = None,
) -> dict[str, str] | None:
	"""Return a clarification payload when the query cannot be answered as-is."""

	normalized_query = str(query_context.get("normalized_query", ""))
	unsupported_filters = query_context.get("unsupported_filters") or []

	if _is_query_too_vague(normalized_query):
		return {
			"reason": "query_too_vague",
			"question": (
				"Your query is too short to determine the analytical goal. "
				"Please specify what you want - for example: segment customers into groups, "
				"show descriptive statistics, or find customers suitable for a product offer."
			),
		}

	if unsupported_filters and not _has_resolvable_intent_context(query_context):
		requested = ", ".join(item["requested"] for item in unsupported_filters)
		return {
			"reason": "unsupported_filters_only",
			"question": (
				f"This dataset does not include: {requested}. "
				"Your query relies on those filters and does not specify another analytical goal. "
				"What would you like to do instead - for example, segment all customers or "
				"show descriptive statistics?"
			),
		}

	if _mentions_threshold_without_anchor(normalized_query):
		return {
			"reason": "threshold_ambiguity",
			"question": (
				"Your query uses a relative value term (such as high value or big spenders) "
				"without a numeric threshold or a known tier name. "
				"Please provide a numeric amount or choose a tier: priority, regular, or dormant."
			),
		}

	return None


def build_context(query: str, dataset_path: str | Path | None = None) -> dict[str, Any]:
	"""Build a structured planner context from a user query."""

	resolved_dataset_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
	normalized_query = normalize_query(query)
	intent_routing = classify_intent_semantically(normalized_query)
	intent = infer_intent(normalized_query, semantic=intent_routing)
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
		intent_routing={
			"method": "tfidf_semantic",
			"category": intent_routing["category"],
			"similarity_score": intent_routing["similarity_score"],
			"matched_example": intent_routing["matched_example"],
			"resolved_intent": intent,
		},
	)
	return context.to_dict()