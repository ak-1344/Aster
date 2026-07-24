# ASTER - Implementation Roadmap

> Version: v0.1
> Goal: Build a fully working agent first. Optimize and scale later.
>
> Rule:
> Every commit must leave ASTER in a runnable state.

---

# Phase 0 - Project Setup

## Goal

Create the project structure.

No AI.

No ML.

No LLM.

No Cache.

No Database.

Just folders and starter files.

### Deliverables

- Backend folder
- Frontend folder
- Module structure
- requirements.txt
- Empty python files
- Git repository

---

# Phase 1 - Core Analytical Nodes

## Goal

Implement every analytical node independently.

No orchestration yet.

Each node should accept a DataFrame and return its own output.

### Tasks

- Dataset Loader
- Analytics Node
- EDA Node
- Feature Engineering Node
- Segmentation Node (KMeans only)
- Recommendation Node (Rule-based)
- Evaluation Node
- Visualization Node

### Output

Each node should work independently.

---

# Phase 2 - Rule-Based Planner

## Goal

Create the first working planner.

No LLM.

Hardcoded routing.

### Example

Query

"Segment customers"

↓

Planner

Feature Engineering

↓

Segmentation

↓

Evaluation

↓

Visualization

---

Query

"Show missing values"

↓

Planner

Analytics

↓

EDA

### Tasks

- Query Analyzer
- Intent Parser
- Planner
- Context Builder

---

# Phase 3 - Execution Engine

## Goal

Create an execution graph.

Planner should no longer directly call nodes.

Instead

Planner

↓

Execution Graph

↓

Scheduler

↓

Nodes

### Tasks

- Workflow Graph
- Task Scheduler
- Response Composer

### Output

ASTER can execute different workflows.

---

# Phase 4 - API & Frontend

## Goal

Create an actual application.

### Backend

- FastAPI
- Endpoints
- Dataset Upload
- Query API

### Frontend

- Chat Interface
- Dataset Upload
- Result Display
- Charts
- Segment Visualization

---

# Phase 5 - LLM Integration

## Goal

Replace the hardcoded planner.

### LLM Usage

Query

↓

LLM

↓

Intent JSON

↓

Planner

Example

User:

"Show dormant customers"

LLM returns

{
    "intent": "SEGMENT",
    "filters": ["Dormant"],
    "output": "table"
}

### Tasks

- Gemini API
- Prompt Engineering
- Intent Parsing

---

# Phase 6 - Explainability

## Goal

Generate human-readable explanations.

### Pipeline

Execution

↓

SHAP / LIME

↓

LLM

↓

Explanation

### Tasks

- SHAP
- LIME
- Decision Engine

---

# Phase 7 - Architecture Expansion

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