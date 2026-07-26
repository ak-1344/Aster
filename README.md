# ASTER — Customer Segmentation & Personalization Agent for Retail Banking

> **Agentic Segmentation Through Execution & Reasoning (ASTER)** transforms natural-language business questions into planned analytical workflows, executes them through modular nodes, and returns explainable segmentation insights and recommendations.

ASTER is an intelligent analytics assistant built for retail banking teams. Instead of running a fixed pipeline for every question or forcing analysts to write complex SQL/Python scripts, ASTER dynamically interprets natural-language queries, plans the required analytical sequence (e.g., data profiling → feature engineering → clustering → recommendation generation → surrogate model explainability), executes the workflow, and translates raw mathematical outputs into human-readable business insights.

*"Think. Plan. Execute. Explain."*

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset Information](#-dataset-information)
- [Solution Approach & Agentic Architecture](#-solution-approach--agentic-architecture)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Usage & API Guide](#-usage--api-guide)
- [AI Tools & Assistance Disclosure](#-ai-tools--assistance-disclosure)
- [Data Sources & Citation](#-data-sources--citation)
- [Documentation Index](#-documentation-index)

---

## 🎯 Problem Statement

Modern retail banking analytics teams face a recurring challenge: bridging the gap between non-technical business decision-makers and complex data science workflows. Business users ask nuanced questions (e.g., *"Identify our high-volume transactors and recommend retention offers"*), but traditional systems force a trade-off between rigid pre-canned dashboards and slow, manual data science cycles.

**ASTER solves Problem Statement 2 (PS2: Customer Segmentation & Personalization Agent for Retail Banking)** by providing an autonomous, agentic system that:
1. Accepts natural-language business queries from retail banking analysts.
2. Dynamically plans and executes multi-step data pipelines (EDA, customer feature engineering, unsupervised clustering, cluster quality evaluation, and personalized product recommendations).
3. Provides full mathematical transparency and surrogate-model explainability (SHAP/LIME) so every customer segment assignment can be audited and understood.

For detailed functional specifications and component boundaries, see [`docs/FEATURES.md`](docs/FEATURES.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 📊 Dataset Information

### Core Dataset: Credit Card Customers (CC GENERAL)
ASTER is configured out of the box for the **Credit Card Customers (CC GENERAL)** dataset, a customer-level behavioral dataset widely used in retail banking analytics.

* **Customer Count:** 8,950 retail credit card customers.
* **Feature Count:** 18 behavioral features (e.g., `BALANCE`, `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, `TENURE`).
* **Storage Path:** `backend/data/raw/CC GENERAL.csv` (gitignored to comply with repository best practices).

### Dataset Sourcing & Bootstrap
To run ASTER locally, you can choose between two dataset provisioning methods:
1. **Synthetic Bootstrap Script (Recommended for testing):**
   Run `python scripts/bootstrap_sample_data.py` to generate a 120-row, schema-compatible synthetic CSV at `backend/data/raw/CC GENERAL.csv` using `numpy.random`.
2. **Real Dataset Placement (For production analytics):**
   Download the original dataset from the UCI Machine Learning Repository and save it to `backend/data/raw/CC GENERAL.csv`.

### Custom Dataset Ingestion (`POST /upload`)
ASTER includes an automated data ingestion and feature engineering endpoint (`POST /upload`). Users can upload any custom CSV file via `multipart/form-data`. The system:
* Validates CSV parseability, row presence, and numeric column eligibility.
* Persists the raw dataset to `backend/data/raw/upload_<timestamp>.csv`.
* Automatically executes the feature engineering pipeline synchronously to generate customer behavioral features at `backend/data/processed/upload_<timestamp>.csv`.
* Returns a `dataset_id` which can be passed directly to subsequent `/query` calls.

---

## 💡 Solution Approach & Agentic Architecture

### Why ASTER is "Agentic" (Hackathon Agentic Criterion)
Per the hackathon's definition of **Agentic AI**, an application is not agentic merely because it invokes a Large Language Model for a single one-shot response. ASTER is genuinely **agentic** because:
- **Dynamic Reasoning & Planning:** The system uses LLM-driven reasoning to interpret query intent and autonomously construct a multi-step DAG (Directed Acyclic Graph) of analytical tasks tailored to the specific query.
- **Tool & Component Orchestration:** The agent dynamically decides *which* analytical components (EDA, feature engineering, clustering algorithms, surrogate explainers) to invoke and in what order.
- **Decoupled Architecture:** Planning (*what to run*), Scheduling (*when to run*), Computation (*how to calculate*), and Explainability (*why it happened*) are strictly decoupled modules.

### Full Request Flow

```text
User Query → FastAPI (/query) 
          → Query Manager (normalization & cache lookup)
          → Context Builder (intent & filter extraction)
          → Planner (Gemini-reasoned graph planning with rule-based fallback)
          → Execution Graph (topological dependency resolution)
          → Task Scheduler (sequential node execution)
          → Analytical Nodes (EDA, Feature Eng, Segmentation, Recommendations, Evaluation, Viz)
          → Decision Engine (Surrogate RF + SHAP/LIME Feature Attribution)
          → Response Composer (unified JSON payload)
```

### Key Modules & Responsibilities
1. **Query Manager & Decision Memory:** Checks SQLite exact-key cache (`backend/data/decision_memory.db`) for near-instant responses on repeated queries.
2. **Context Builder:** Normalizes input text and extracts parameters (e.g., target cluster count, customer IDs).
3. **Planner:** Prompts Google Gemini to output a structured JSON plan selecting required nodes from the component catalog. Automatically falls back to deterministic rule templates if LLM services are unavailable or time out (10s limit).
4. **Model Registry:** Provides dynamic factory functions for clustering algorithms (`KMeans`, `DBSCAN`, `HDBSCAN`). Gemini assists in algorithm selection, while `scikit-learn` handles deterministic model fitting.
5. **Decision Engine & Explainability:** Fits a surrogate `RandomForestClassifier` on cluster labels and executes SHAP (`TreeExplainer`) or LIME (`TabularExplainer`) based on the `EXPLAINABILITY_MODE` environment variable, with centroid-distance rule fallback.
6. **Response Composer:** Assembles segment profiles, personalized product recommendations, visual scatter data, and execution logs into a clean response schema.

---

## 🛠️ Tech Stack

### Production Stack (Currently Built & Active)
* **API & Server Framework:** [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) (Asynchronous ASGI server with Pydantic validation)
* **Data Processing & Analytics:** [pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
* **Machine Learning & Clustering:** [scikit-learn](https://scikit-learn.org/) (KMeans, DBSCAN, HDBSCAN, RandomForest surrogate models)
* **Explainable AI (XAI):** [SHAP](https://shap.readthedocs.io/) & [LIME](https://lime-ml.readthedocs.io/)
* **LLM Reasoning & Orchestration:** [Google Gemini API](https://ai.google.dev/) (`google-genai` SDK with structured JSON schemas and timeout controls)
* **Data Visualization:** [Plotly](https://plotly.com/python/) (Scatter plot data structure generation)
* **Caching & Decision Memory:** SQLite3 (`backend/data/decision_memory.db`)
* **Environment & HTTP Utilities:** `python-dotenv`, `python-multipart` (CSV file uploads), `httpx2` (test client integration)
* **Frontend:** Vanilla HTML5 / JavaScript / CSS single-page interface served statically via FastAPI at `/`

### Planned Future Work (Out of Scope for Current MVP)
* **Frontend Migration:** Next.js + TailwindCSS modern reactive dashboard UI.
* **Vector Store / Semantic Cache:** Upgrading exact-key SQLite cache to FAISS or Redis for embedding-based semantic similarity query caching.
* **Authentication & Multi-Tenancy:** OAuth2 / JWT authentication and user session management.

---

## ⚙️ Setup & Installation

Follow these canonical steps to install, configure, and execute ASTER locally.

### 1. Prerequisites
* **Python:** 3.11 or higher
* **Git**
* **Google Gemini API Key:** Required for LLM planning (optional; fallback mode operates cleanly without it).

### 2. Repository Cloning & Environment Setup
```bash
# Clone the repository
git clone https://github.com/ak-1344/Aster---Distributed-Thinking.git
cd Aster---Distributed-Thinking

# Create and activate a virtual environment
# Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Dependency Installation
```bash
pip install -r backend/requirements.txt
```

### 4. Environment Configuration
Copy the `.env.example` file to create your local `.env`:
```bash
cp .env.example .env
```
Edit `.env` to configure your environment:
```ini
# Required for LLM reasoning (system uses rule-based fallback if omitted)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional configuration controls
EXPLAINABILITY_MODE=shap       # Options: shap, lime, rule_based (default: shap)
GEMINI_TIMEOUT_SECONDS=10       # LLM response timeout threshold
GEMINI_MODEL=gemini-2.5-flash  # Target Gemini model name
```

### 5. Dataset Provisioning
Generate the synthetic demo dataset:
```bash
python scripts/bootstrap_sample_data.py
```
*(Or place your real `CC GENERAL.csv` at `backend/data/raw/CC GENERAL.csv`)*

### 6. Run the Application
Start the unified FastAPI server from the **repository root**:
```bash
python backend/main.py
```
* **Interactive Dashboard:** Open `http://127.0.0.1:8000/` in any browser.
* **API Documentation:** Open `http://127.0.0.1:8000/docs` for Swagger UI.

### 7. Run the Test Suite
Execute the full smoke and unit test suite (49 tests):
```bash
python -m unittest discover -s backend/tests -p "test_*.py" -v
```

---

## 📖 Usage & API Guide

### 1. Web Interface
Access `http://127.0.0.1:8000/` to use the embedded natural-language query interface. Enter queries like *"segment customers into 3 clusters"* to view execution plans, persona breakdowns, product recommendations, surrogate SHAP explanations, and interactive data tables.

### 2. Natural Language Query Endpoint (`POST /query`)

#### Example A: Full Customer Segmentation
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "segment customers into 3 clusters based on spending behavior"}'
```

#### Example B: EDA / Descriptive Statistics Only
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "show descriptive statistics for the dataset"}'
```

#### Example C: Explanation-Only Follow-Up
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Why was customer 1004 placed in their current segment?"}'
```

#### Example D: Targeting a Custom Dataset ID
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "segment customers into 3 clusters", "dataset_id": "upload_20260726_120000"}'
```

### 3. Custom Dataset Upload Endpoint (`POST /upload`)
Upload a new CSV file to trigger automated feature engineering and generate a reusable `dataset_id`:
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@/path/to/your/custom_dataset.csv"
```
**Response:**
```json
{
  "status": "success",
  "rows_ingested": 500,
  "features_generated": 13,
  "dataset_id": "upload_20260726_120000",
  "preview": [...]
}
```

---

## 🤖 AI Tools & Assistance Disclosure

In compliance with hackathon regulations regarding external tool disclosure:

1. **AI Assistance Scope:** Multiple AI coding tools and LLM APIs were used throughout the development of ASTER. These tools assisted across multiple phases, including architecture planning, module scaffolding, code implementation, test suite writing, debugging, and technical documentation drafting.
2. **LLM Runtime Integration:** The core system incorporates Google Gemini API as an intelligent component for natural-language query planning, algorithm parameter suggestion, feature relevance ranking, and business narrative phrasing.
3. **Human Control & Originality:** All architectural boundaries, module design decisions, business rules, deterministic fallback paths, scope limits, and final code verification were strictly authored, reviewed, and controlled by the human engineering team. The code strictly adheres to project design constraints, ensuring reproducible, deterministic analytical outputs.

---

## 📚 Data Sources & Citation

### Credit Card Customers (CC GENERAL) Dataset
* **Dataset Title:** Credit Card Dataset for Data Mining / Default of Credit Card Clients
* **Source Repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **Citation:** 
  > Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card customers. *Expert Systems with Applications*, 36(2), 2473-2480.
* **License & Usage:** Public Domain / Available for academic, research, and technical demonstration purposes.

---

## 📁 Documentation Index

Detailed architectural and workflow documentation is maintained in the `docs/` folder:

* 🏛️ **[Architecture Overview](docs/ARCHITECTURE.md)**: In-depth module responsibilities, request flow DAG, model registry, and decision memory design.
* ✨ **[Features Guide](docs/FEATURES.md)**: Functional breakdown of EDA, segmentation, persona generation, surrogate explainability, and LLM vs. deterministic boundaries.
* 🔄 **[User Workflow Guide](docs/USER_WORKFLOW.md)**: End-to-end walkthroughs of EDA queries, full segmentation, explanation follow-ups, and cache-hit mechanics.
* ⚙️ **[Setup & Installation Guide](docs/SETUP.md)**: Canonical setup, environment variables, bootstrap scripts, and testing instructions.

---

*"Think. Plan. Execute. Explain."*
