# 📋 ASTER — ToDo

> Maintained by agent. Updated after every task.
> Pull phase details from plan.md.

---

## ✅ Completed Milestone: Phase 0 — Project Setup

- [x] `git init` at repo root
- [x] Add `.gitignore` (`__pycache__/`, `.venv/`, `*.pyc`, `node_modules/`, `.next/`, `*.env`)
- [x] Initial commit: `chore: project scaffold`
- [x] Create `backend/app/api/__init__.py`, `backend/app/api/main.py` (empty `FastAPI()` instance, no routes)
- [x] Create `backend/app/query_manager/__init__.py`, `query_manager.py`
- [x] Create `backend/app/context_builder/__init__.py`, `context_builder.py`
- [x] Create `backend/app/planner/__init__.py`, `planner.py`
- [x] Create `backend/app/execution_graph/__init__.py`, `execution_graph.py`
- [x] Create `backend/app/scheduler/__init__.py`, `scheduler.py`
- [x] Create `backend/app/response_composer/__init__.py`, `response_composer.py`
- [x] Create `backend/app/nodes/__init__.py`, `analytics_node.py`, `eda_node.py`, `feature_engineering_node.py`, `segmentation_node.py`, `recommendation_node.py`, `evaluation_node.py`, `visualization_node.py`
- [x] Create `backend/app/decision_engine/__init__.py`, `decision_engine.py`
- [x] Every `.py` file above: docstring only, no logic beyond `pass`
- [x] Create `backend/data/raw/`, `backend/data/processed/`
- [x] Place `CC GENERAL.csv` in `backend/data/raw/`
- [x] Create `frontend/README.md` placeholder only — do **not** run `create-next-app` (Frontend is Phase 7)
- [x] Create `backend/requirements.txt`: `fastapi`, `uvicorn[standard]`, `pandas`, `numpy`, `scikit-learn`, `shap`, `lime`, `plotly`, `google-generativeai`, `python-dotenv`, `pydantic`
- [x] Create `.env.example` (`GEMINI_API_KEY=`, no real key committed)
- [x] `python -m venv .venv`
- [x] `pip install -r backend/requirements.txt`

### Exit Criteria

- [x] `pip install -r backend/requirements.txt` completes clean in a fresh venv
- [x] `python -c "import app.api.main"` runs with no import errors
- [x] `git log` shows at least one commit
- [x] Every file listed above exists and is syntactically valid Python

---

## ✅ Completed Milestone: Phase 1 — Dataset Understanding

- [x] Inspect `backend/data/raw/CC GENERAL.csv`
- [x] Analyze missing values
- [x] Analyze datatypes
- [x] Generate statistical summary
- [x] Identify customer identifier
- [x] Identify transaction identifier
- [x] Create dataset-understanding report documentation
- [x] Add reusable dataset profile utility
- [x] Add dataset-understanding node wrapper

### Exit Criteria

- [x] Dataset schema is documented
- [x] Identifier fields are documented
- [x] Missingness is documented
- [x] Dataset-understanding code validates successfully

---

## ✅ Completed Milestone: Phase 2 — Feature Engineering

- [x] Implement reusable customer feature engineering utility
- [x] Implement feature engineering node wrapper
- [x] Generate `backend/data/processed/customer_features.csv`
- [x] Handle missing numeric values before feature computation
- [x] Add safe ratio computations for engineered features
- [x] Document feature engineering outputs and constraints

### Exit Criteria

- [x] `backend/data/processed/customer_features.csv` exists
- [x] Generated feature file has one row per customer
- [x] Feature engineering module compiles and executes without import errors
- [x] Engineered features are documented

---

## ✅ Completed Milestone: Phase 3 — Analytical Nodes

- [x] Implement analytics node with descriptive stats API
- [x] Expand EDA node outputs for reusable exploratory summaries
- [x] Connect feature engineering node output into node-level workflow contract
- [x] Implement segmentation node baseline (KMeans)
- [x] Implement recommendation node baseline (rule-based)
- [x] Implement evaluation node baseline (silhouette score and cluster size summary)
- [x] Implement visualization node baseline outputs
- [x] Add independent node-level smoke tests for each node
- [x] Add temporary FastAPI workflow endpoint for direct node execution
- [x] Add temporary frontend demo page for workflow execution

### Exit Criteria

- [x] Each node can run independently with local inputs
- [x] Segmentation output includes cluster labels per customer
- [x] Evaluation output includes at least one clustering quality metric
- [x] Node smoke tests pass
- [x] Temporary demo path runs end-to-end through the analytical nodes

## ✅ Completed Milestone: Phase 4 — Planner

- [x] Implement rule-based planner entrypoint in `backend/app/planner/planner.py`
- [x] Add query normalization and intent routing in `backend/app/query_manager/query_manager.py`
- [x] Add context-building support for planner inputs in `backend/app/context_builder/context_builder.py`
- [x] Add planner smoke tests for segmentation and descriptive queries

### Exit Criteria

- [x] Planner returns an executable workflow for at least one analytical query
- [x] Planner smoke tests pass
- [x] Planner uses the Phase 3 analytical nodes as execution targets

## ✅ Completed Milestone: Phase 5 — Execution Engine

- [x] Implement execution graph data structures in `backend/app/execution_graph/execution_graph.py`
- [x] Implement scheduler execution in `backend/app/scheduler/scheduler.py`
- [x] Implement response composition in `backend/app/response_composer/response_composer.py`
- [x] Add execution-engine smoke tests for workflow assembly and response composition

### Exit Criteria

- [x] Planner output can be consumed by the execution graph and scheduler
- [x] Execution-engine smoke tests pass
- [x] Workflow responses combine statistics, recommendations, and visual outputs

## 🟡 Current Phase: Phase 6 — API Layer (Blocking)

- [ ] Implement `POST /upload` endpoint in `backend/app/api/main.py` for dataset upload
- [ ] Implement `POST /query` endpoint in `backend/app/api/main.py` for natural language analytical requests
- [ ] Implement `GET /status` endpoint in `backend/app/api/main.py` for system health and execution status
- [ ] Wire `/query` endpoint through Query Manager → Planner → Execution Graph → Scheduler → Response Composer
- [ ] Remove or replace temporary `/run-workflow` endpoint (per ISSUE-002)
- [ ] Add API smoke tests for endpoint validation
- [ ] Add request/response Pydantic models for type safety

### Exit Criteria

- [ ] `/upload` endpoint accepts and validates dataset files
- [ ] `/query` endpoint returns structured responses from the execution engine
- [ ] `/status` endpoint returns system health information
- [ ] API smoke tests pass
- [ ] Temporary `/run-workflow` endpoint is removed or replaced with proper API integration

## Backlog — Phase 7 onward

- [ ] Phase 7 — Frontend
- [ ] Phase 8 — LLM Integration
- [ ] Phase 9 — Explainability
- [ ] Phase 10 — Advanced Features (Cache, Model Registry, Decision Memory, Auth, Session Management, Metrics)
