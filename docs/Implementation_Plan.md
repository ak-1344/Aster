# ASTER - Implementation Roadmap

> Version: v1.0
> Status: All Phases (0 through 10) completed successfully.
> Alignment note: this roadmap reflects the fully implemented repository state, test suite coverage, and active feature set.

---

## Repository-aligned milestone status

- Phase 0 — Project Setup: Completed. Scaffold, environment isolation, unit testing harness, and dependency specification.
- Phase 1 — Dataset Understanding: Completed. Automated profiling, missing-value imputation, schema validation, and customer-level domain mapping.
- Phase 2 — Feature Engineering: Completed. Customer behavioural ratios, spend metrics, and processed dataset caching.
- Phase 3 — Analytical Nodes: Completed. Independent modular nodes (Analytics, EDA, Feature Engineering, Segmentation, Recommendation, Evaluation, Visualization).
- Phase 4 — Planner: Completed. Intent extraction, query context parsing, parameter extraction, and unsupported filter detection.
- Phase 5 — Execution Engine: Completed. Directed Execution Graph building, sequential Task Scheduler with timing telemetry, and Response Composer.
- Phase 6 — API Layer: Completed. FastAPI application exposing `/query`, `/upload`, `/dashboard/node-stats`, `/dashboard/graph`, and `/dashboard/live` WebSocket telemetry.
- Phase 7 — Frontend UI: Completed. Neumorphic Banking Analytics Dashboard with real-time execution trace, interactive charts, unsupported filter alerts, and dataset upload interface.
- Phase 8 — LLM Integration: Completed. Gemini-driven planner, model registry parameter selection, feature relevance ranking, and natural-language recommendation synthesis with deterministic fallback.
- Phase 9 — Explainability: Completed. SHAP & LIME surrogate models for cluster interpretation, distance-based rule-based fallback, and `EXPLAINABILITY_MODE` toggle.
- Phase 10 — Advanced Features: Completed. SQLite Decision Memory exact-key caching, Model Registry (KMeans, DBSCAN, HDBSCAN), WebSocket Pub/Sub event bus, node timing telemetry, and custom CSV upload pipeline.

---

## Phase 0 - Project Setup

### Status
Completed.

### Delivered
- Backend and frontend package scaffolds.
- Module structure for nodes, planner, scheduler, workflow, decision memory, and utilities.
- Dependency manifest (`backend/requirements.txt`) and environment bootstrap files (`.env.example`).
- FastAPI entrypoint in `backend/main.py` with automatic `sys.path` resolution.

---

## Phase 1 - Dataset Understanding

### Status
Completed.

### Delivered
- Dataset loading and profiling utilities in `backend/app/query_manager/dataset_manager.py`.
- Missing-value, datatype, and identifier analysis.
- Domain documentation for the Credit Card Customers dataset (`CC GENERAL.csv`).
- Fast synthetic bootstrap script (`scripts/bootstrap_sample_data.py`).

---

## Phase 2 - Feature Engineering

### Status
Completed.

### Delivered
- 11 customer-level engineered behavioral features (`monthly_spend`, `credit_utilization_ratio`, `cash_advance_ratio`, `payment_to_minimum_ratio`, `credit_headroom`, etc.).
- Safe division and median imputation for numerical stability.
- Processed feature artifact written to `backend/data/processed/customer_features.csv`.

---

## Phase 3 - Analytical Nodes

### Status
Completed.

### Delivered
- **Analytics Node:** Computes descriptive statistical summaries across dataset features.
- **EDA Node:** Generates exploratory distribution statistics and missing value profiles.
- **Feature Engineering Node:** Computes customer behavioral ratios on-the-fly.
- **Segmentation Node:** Executes customer clustering via active model registry factories.
- **Recommendation Node:** Ranks and pairs targeted financial action items with customer segments.
- **Evaluation Node:** Reports Silhouette Score and cluster size distributions.
- **Visualization Node:** Produces 2D/3D scatter payloads for interactive rendering.
- Complete unit test suite verifying each node independently.

---

## Phase 4 - Planner

### Status
Completed.

### Delivered
- Structured intent classification (`descriptive_eda`, `segmentation_workflow`, `explanation_only`).
- Context builder (`context_builder.py`) extracting requested cluster count, filter rules, and entities.
- Unsupported filter detection mechanism comparing requested criteria against available dataset schema.
- Deterministic rule-based fallback planner when LLM key is absent or unreachable.

---

## Phase 5 - Execution Engine

### Status
Completed.

### Delivered
- `ExecutionGraph` class for topological node dependency resolution.
- `Scheduler` for step-by-step sequential node execution with node-level execution timing logs.
- `ResponseComposer` merging node outputs into unified, standard JSON contracts.

---

## Phase 6 - API Layer

### Status
Completed.

### Delivered
- `POST /query`: Primary endpoint processing natural-language business queries.
- `POST /upload`: Custom dataset ingest endpoint validating CSV schemas and triggering immediate feature engineering.
- `GET /dashboard/node-stats`: Exposes real-time node performance metrics.
- `GET /dashboard/graph`: Serves active query execution topology.
- `WebSocket /dashboard/live`: In-memory Pub/Sub channel for live execution event streaming.

---

## Phase 7 - Frontend

### Status
Completed.

### Delivered
- Neumorphic & glassmorphic dark-mode UI (`frontend/index.html` & `frontend/dashboard.html`).
- Interactive Chart.js visualizations for cluster breakdowns and feature distributions.
- Live execution progress bar and real-time step updates via WebSockets.
- Unsupported query filter alerts to keep analysts informed of query constraints.
- Drag-and-drop CSV dataset upload interface.

---

## Phase 8 - LLM Integration

### Status
Completed.

### Delivered
- Structured JSON-schema Gemini prompts for Intent Planning and Algorithm Recommendation.
- Dynamic feature relevance ranking guiding clustering towards intent-focused signals.
- LLM-assisted recommendation phrasing translating rigid rule outputs into executive summaries.
- Robust exception handling guaranteeing 100% execution continuity via deterministic fallback paths under API timeouts or rate limits.

---

## Phase 9 - Explainability

### Status
Completed.

### Delivered
- SHAP (`TreeExplainer` / `KernelExplainer`) surrogate model explanations for segment attribution.
- LIME (`TabularExplainer`) surrogate model alternative.
- Distance-to-centroid rule-based fallback for high-speed or single-cluster edge cases.
- `EXPLAINABILITY_MODE` environment variable toggle (`shap`, `lime`, `rule_based`).

---

## Phase 10 - Advanced Features

### Status
Completed.

### Delivered
- **Decision Memory Cache:** SQLite-backed exact-key caching (`backend/data/decision_memory.db`) eliminating redundant compute and LLM calls for identical queries.
- **Model Registry:** Pluggable model factories for `KMeans`, `DBSCAN`, and `HDBSCAN`.
- **Telemetry Event Bus:** Real-time event broadcasting during node execution for live system monitoring.
- **Custom Dataset Management:** Multi-dataset query isolation and session-less custom dataset processing.

---

# Future Roadmap & Architectural Opportunities

- **Semantic Query Caching:** Embedding-based query similarity matching.
- **Parallel Node Execution:** Multi-threaded graph execution for independent nodes.
- **PostgreSQL / Redis Storage:** Production-grade persistence replacing SQLite for high-concurrency deployments.
- **User Authentication & Session Management:** Multi-tenant access controls and query history per user session.