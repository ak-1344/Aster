# ASTER - Implementation Roadmap

> Version: v0.3
> Status: Phase 3 completed; Phase 4 planner is the current milestone.
> Alignment note: this roadmap reflects the repository history and the current working tree rather than the earlier placeholder plan.

---

## Repository-aligned milestone status

- Phase 0 — Project Setup: completed in the repository scaffold and initial backend package layout.
- Phase 1 — Dataset Understanding: completed with the dataset loading utility, schema report, and documentation.
- Phase 2 — Feature Engineering: completed with reusable customer-feature generation and the processed CSV artifact.
- Phase 3 — Analytical Nodes: completed with runnable analytics, EDA, segmentation, recommendation, evaluation, and visualization nodes.
- Phase 4 — Planner: current milestone for rule-based routing and workflow selection.
- Phase 5 onward: execution engine, API layer, frontend, LLM integration, explainability, and advanced features.

---

## Phase 0 - Project Setup

### Status
Completed.

### Delivered
- Backend and frontend package scaffolds
- Module structure for nodes, planner, scheduler, workflow, and utilities
- Dependency manifest and environment bootstrap files
- Initial FastAPI entrypoint and placeholder modules

### Commit alignment
The repository history includes the initial scaffold and Phase 0 setup work before the later dataset and feature-engineering milestones.

---

## Phase 1 - Dataset Understanding

### Status
Completed.

### Delivered
- Dataset loading and profiling utilities
- Missing-value, datatype, and identifier analysis
- Documentation for the selected customer dataset

### Commit alignment
This phase was implemented in the repository history around the first dataset-understanding milestone.

---

## Phase 2 - Feature Engineering

### Status
Completed.

### Delivered
- Customer-level engineered features
- Processed feature artifact written to backend/data/processed/customer_features.csv
- Documentation of engineered features and assumptions

### Commit alignment
The current repository history includes the completed feature-engineering milestone and the generated feature dataset.

---

## Phase 3 - Analytical Nodes

### Status
Completed.

### Delivered
- Analytics node with descriptive statistics output
- EDA node with reusable exploratory summaries
- Segmentation node using a KMeans baseline
- Recommendation node using simple rule-based ranking
- Evaluation node reporting silhouette score and cluster sizes
- Visualization node producing a lightweight scatter payload
- Node smoke tests covering each analytical component

### Execution note
Each node operates independently and can be exercised against the current customer dataset and generated feature file.

---

## Phase 4 - Planner

### Status
In progress.

### Planned work
- Implement a rule-based planner entrypoint
- Normalize user queries into intent and workflow selections
- Build a lightweight context object from the dataset and engineered features
- Connect the planner to the Phase 3 node outputs

### Expected outcome
The planner should be able to route a natural-language request such as “segment customers” into an executable sequence of feature engineering, segmentation, evaluation, and visualization steps.

---

## Phase 5 - Execution Engine

### Status
Planned.

### Planned work
- Introduce an execution graph abstraction
- Coordinate node execution through a scheduler
- Compose a response payload for downstream API consumption

---

## Phase 6 - API Layer

### Status
Planned.

### Planned work
- Expose FastAPI endpoints for dataset upload and query execution
- Return structured analytics responses to the frontend layer

---

## Phase 7 - Frontend

### Status
Planned.

### Planned work
- Add a lightweight chat-style interface
- Display analytical outputs and charts
- Support basic dataset upload and result rendering

---

## Phase 8 - LLM Integration

### Status
Planned.

### Planned work
- Replace or augment the rule-based planner with Gemini-powered intent parsing
- Keep the analytical core independent from LLM orchestration

---

## Phase 9 - Explainability

### Status
Planned.

### Planned work
- Add SHAP or LIME-based explanations for segmentation outputs
- Produce human-readable summaries for key results

---

## Phase 10 - Advanced Features

### Status
Planned.

### Planned work
- Cache execution results
- Introduce model registry and decision memory concepts
- Add authentication, session management, and basic metrics

## Goal

Implement the remaining architecture components.

### Components

- Cache
- Model Registry
- Decision Memory
- Authentication
- Session Manager
- Metrics
- Logging

Each feature should be implemented one at a time.

ASTER should remain runnable after every feature.

---

# Future Improvements

- DBSCAN
- HDBSCAN
- Automatic Model Selection
- Better Recommendation Engine
- Better Prompt Templates
- Decision Memory Optimization
- Distributed Execution
- Parallel Node Execution
- Performance Optimization

---

# Development Rules

## Rule 1

Never implement multiple major components together.

---

## Rule 2

Every commit must leave ASTER runnable.

---

## Rule 3

Finish one phase before starting the next.

---

## Rule 4

Keep architecture fixed.

Do not redesign unless absolutely necessary.

---

## Rule 5

Optimization comes after correctness.

A working simple system is always better than an unfinished complex system.