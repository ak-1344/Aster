# ASTER Demo Script for Judges

Use this script at `http://127.0.0.1:8000/` or via `POST /query`. Each query demonstrates a distinct pipeline behavior introduced or fixed across recent milestones.

## Prerequisites

From the repository root:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r backend/requirements.txt
python scripts/bootstrap_sample_data.py
python backend/main.py
```

---

## Query 1 — Plain descriptive statistics

**Query:** `show descriptive statistics for the dataset`

**What to notice:**
- `workflow_name` is `descriptive_workflow`
- `metadata.nodes_executed` chips show `analytics` and `eda` only
- No segmentation, recommendations, or cluster explanations
- Statistics panel populates with numeric and exploratory summaries

---

## Query 2 — Keyword segmentation

**Query:** `segment customers into 3 clusters`

**What to notice:**
- Full segmentation workflow: feature engineering → segmentation → evaluation → recommendation → visualization
- Node chips list all five downstream nodes
- Cluster recommendations table includes tier tags and behavioral product tags where rules match
- Silhouette score appears in the Evaluation panel

---

## Query 3 — Business-phrased segmentation (ISSUE-007)

**Query:** `Find customers suitable for investment products`

**What to notice:**
- No cluster/segment vocabulary in the query, yet ASTER still routes to segmentation
- `intent` is `segmentation`; nodes_executed includes `segmentation` and `recommendation`
- Demonstrates `BUSINESS_INTENT_KEYWORDS` routing in Context Builder

---

## Query 4 — Clarification trigger (ISSUE-008)

**Query:** `high value customers`

**What to notice:**
- Response returns `status: clarification_needed` (not a full workflow)
- UI shows a clarification panel instead of results tables
- Question asks for a numeric threshold or tier name (`priority`, `regular`, `dormant`)
- No nodes execute; decision memory is not written

**Follow-up (optional):** submit clarification `segment customers into 3 clusters` to run the full pipeline.

---

## Query 5 — Unsupported filter with resolvable intent

**Query:** `Show premium customers from Chennai.`

**What to notice:**
- ISSUE-009 fix: `"show"` no longer misroutes to `explanation_only` via substring `"how"`
- `premium` triggers segmentation intent despite Chennai being unsupported
- Amber notice lists dropped filter `city/location`
- Pipeline still executes segmentation; results reflect all customers (filter not applied)

---

## Query 6 — Genuine explanation-only query

**Query:** `explain why this customer is in this segment`

**What to notice:**
- `intent_classification` is `explanation_only`
- `metadata.nodes_executed` is empty — no analytical nodes re-run
- Appropriate for follow-up questions after a prior segmentation query populated decision memory

---

## Bonus — Borderline customer signal (Task B)

After any segmentation query, open **Customer Explanations** and look for rows where:
- `boundary_distance_ratio` is present (numeric, 0–1 range typical)
- Explanation text includes *"This customer is borderline between Segment X and Segment Y"* when ratio exceeds 0.85

This shows near-threshold cluster assignments without changing the underlying KMeans labels.
