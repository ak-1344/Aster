# ASTER Features Guide

This document outlines the current capabilities of the ASTER platform from a user's perspective. ASTER relies on natural-language queries rather than static dashboards; it interprets what you want and dynamically builds a workflow to provide the insights.

## Core Capabilities

### 1. Automated Exploratory Data Analysis (EDA)
**What it does:** Automatically profiles the dataset, identifies missing values, uncovers statistical distributions, and summarises key behavioural traits of your customer base. It transforms raw transactions into aggregated customer-level features (e.g., monthly spend, transaction frequency).
**What triggers it:** Descriptive queries or requests for dataset summaries.
**Example query:** *"Show me the descriptive statistics for our customer dataset."*

### 2. Customer Segmentation
**What it does:** Groups customers into distinct clusters based on their engineered behavioural features. Instead of relying on a hardcoded algorithm, ASTER dynamically selects the best approach (KMeans, DBSCAN, or HDBSCAN) and parameters for your specific query.
**What triggers it:** Direct segmentation requests or queries asking to group customers.
**Example query:** *"Segment our customers into 3 clusters based on their spending behaviour."*

### 3. Persona Generation
**What it does:** Synthesises the technical output of the segmentation into human-readable business personas. It looks at the defining features of each cluster (e.g., high credit utilisation, high cash advances) and provides a clear summary of who makes up that segment.
**What triggers it:** Runs automatically alongside any segmentation query.
**Example query:** *"Divide the dataset into 4 customer groups and describe their profiles."*

### 4. Personalised Recommendations
**What it does:** Suggests specific business actions or financial products tailored to each customer segment based on a rule-engine. It translates rigid rule logic into natural, human-readable narratives.
**What triggers it:** Runs automatically for segmentation queries.
**Example query:** *"Group my users into 3 clusters and tell me what products to recommend to them."*

### 5. Explainability (Why were they grouped?)
**What it does:** Explains the reasoning behind every customer's segment assignment and the overarching traits of the segment.
**The approach:** Because clustering is an "unsupervised" machine learning task (it doesn't have ground truth labels), ASTER builds a lightweight "surrogate model" to predict the clusters it just created. It then uses industry-standard frameworks (SHAP or LIME) on this surrogate to extract the true feature importance for each customer.
**The EXPLAINABILITY_MODE switch:** Administrators can toggle how ASTER explains decisions by changing this environment variable to `shap`, `lime`, or `rule_based`. If SHAP or LIME takes too long or fails, ASTER automatically falls back to a fast, reliable rule-based explanation based on the customer's distance from the cluster's centre.
**What triggers it:** Runs alongside segmentation queries, or can be triggered independently as a follow-up.
**Example query:** *"Why was customer 1004 placed in their current segment?"*

### 6. Cluster Evaluation
**What it does:** Assesses the mathematical quality of the generated segments using metrics like the Silhouette Score. This helps analysts gauge if the clusters are distinct and cohesive.
**What triggers it:** Runs automatically for segmentation queries.
**Example query:** *"Segment the customers into 3 groups and evaluate the quality of the clusters."*

### 7. Visualization
**What it does:** Generates the underlying data required to render interactive charts (like scatter plots coloured by cluster) in the frontend.
**What triggers it:** Any query that modifies or summarises data.
**Example query:** *"Segment customers into 3 groups and show me the visualisations."*

### 8. Live Execution Telemetry & Metrics Dashboard
**What it does:** Provides a real-time view into the agent's thought process and execution pipeline. As ASTER plans and executes nodes (e.g. feature_engineering -> segmentation -> recommendation), the progress is broadcast live via WebSockets. A dedicated dashboard visualises node execution times, success rates, and the topological graph of the query.
**What triggers it:** Opening the dashboard or running a query on the main interface.
**Example query:** (No query needed, monitors all queries on the system live).

### 9. Custom Dataset Upload & Auto Feature-Engineering
**What it does:** Allows users to drag and drop custom customer CSV datasets via the frontend or `POST /upload`. ASTER validates column types, imputes missing values, and automatically generates behavioral feature ratios for immediate querying.
**What triggers it:** Dragging a CSV file into the upload zone or invoking `POST /upload`.

### 10. Unsupported Filter Detection & Surfacing
**What it does:** Compares user-requested search filters (e.g., "customers aged over 30 in New York") against the actual dataset schema. If filters refer to attributes missing from the dataset, ASTER surfaces a clear alert banner to the user while gracefully continuing execution on available behavioral data.
**What triggers it:** Queries containing filters for non-existent columns.
**Example query:** *"Segment high spenders aged 25 to 35 in London."*

### 11. Instant Decision Memory Caching
**What it does:** Caches executed analytical plans and outputs in a SQLite database (`backend/data/decision_memory.db`). Repeated queries bypass LLM and clustering latency, returning stored results instantaneously with cache metadata flags.
**What triggers it:** Submitting an identical query twice.

### 12. Query Clarification & Human-in-the-Loop (Active Learning)
**What it does:** Prevents the system from making dangerous assumptions on ambiguous queries. If a query is too vague (e.g., just saying "high value customers" without a numeric threshold or known tier), ASTER immediately halts execution. It returns a `clarification_needed` status and asks the user for clarification. No execution nodes are triggered, and no incorrect assumptions pollute the decision memory.
**What triggers it:** Ambiguous queries lacking explicit intent, unsupported filter-only queries, or missing numeric thresholds.
**Example query:** *"high value customers"*

### 13. TF-IDF Semantic Intent Routing
**What it does:** Provides robust, offline natural-language understanding for queries that don't use exact keywords. Instead of brittle token matching, ASTER uses a Scikit-learn TF-IDF vectorizer to find semantic similarity against labeled intent examples. This ensures accurate routing even when the Gemini API is disabled.
**What triggers it:** Any query processed by the Context Builder.

### 14. Persona Catalog & Product Recommendation Tools
**What it does:** Integrates a fully structured persona catalog (`The Transactors`, `The Revolvers`, `Cash-Advance Reliant`, `Dormant / Low Activity`) directly into the segmentation toolset (`backend/app/tools/segmentation.py`). It automatically maps clusters to these personas and pairs them with specific retail banking products, providing targeted marketing advice.
**What triggers it:** Segmentation workflows mapping customers to business recommendations.

### 15. Borderline Customer Signal
**What it does:** Detects when a customer is mathematically stranded between two clusters. It calculates a `boundary_distance_ratio` to flag cases where a customer is a borderline fit (ratio > 0.85). The explanations explicitly surface this nuance, noting when a customer is "borderline between Segment X and Segment Y".
**What triggers it:** Customer explanation requests or reading explanations for top customers in a cluster.

### 16. Structured Markdown Output Formats
**What it does:** Ensures consistent, highly readable agent answers. Responses are formatted into a strict four-section Markdown structure: (1) Query-Aware Execution Summary, (2) Primary Finding & Persona Match, (3) Target Customer Table, and (4) Strategic Marketing Recommendation.
**What triggers it:** All analytical workflows return this structured Markdown payload.

---

## LLM Integration: What's AI vs What's Deterministic?

ASTER heavily integrates Large Language Models (specifically Google's Gemini), but it maintains a strict architectural boundary between LLM "reasoning" and deterministic "computation."

### What Gemini Reasons About (AI-Driven)
*   **Query Intent & Node Sequencing:** Determining exactly which analytical steps (nodes) are needed to answer a user's question, and in what order.
*   **Algorithm & Parameter Selection:** Looking at the query and the active model registry to choose the best algorithm (e.g. DBSCAN instead of KMeans) and appropriate parameters.
*   **Feature Relevance Ranking:** Ranking the existing engineered features based on the query intent so the clustering algorithm focuses on the right signals.
*   **Recommendation Phrasing:** Taking the strict output of the rule-engine and rephrasing it into a natural, persuasive business narrative.

### What Stays Deterministic (Rule-Based & Math-Driven)
*   **Feature Engineering:** The actual computation of customer features (e.g. calculating averages, handling missing data) is strictly code-driven. The LLM never computes or alters the data.
*   **Clustering Computation:** The actual execution of KMeans, DBSCAN, or HDBSCAN is performed by `scikit-learn`. The LLM only *suggests* the algorithm; it does not calculate the clusters.
*   **Recommendation Logic:** The underlying decision of *which* product to recommend is strictly governed by the rule-engine. The LLM only translates the output into business language.
*   **Pipeline Execution:** The Task Scheduler strictly executes the topological graph in dependency order. The LLM cannot reorder or skip dependencies.

This division ensures that while ASTER is highly flexible and intelligent in *understanding* a request, its mathematical and business logic remains entirely transparent, reproducible, and explainable.
