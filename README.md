# ASTER — Customer Segmentation & Personalization Agent for Retail Banking

> **Agentic Segmentation Through Execution & Reasoning (ASTER)** transforms natural-language business questions into planned analytical workflows, executes them through modular nodes, and returns explainable segmentation insights and recommendations.

---

## Project Overview

ASTER is an intelligent analytics assistant for retail banking teams. Instead of running a fixed pipeline for every question, ASTER:

1. **Understands** the user query (intent, filters, entities)
2. **Plans** which analytical nodes to run
3. **Executes** them in dependency order via a scheduler
4. **Explains** segment assignments with rule-based narratives
5. **Presents** recommendations, statistics, and chart-ready data

This hackathon build focuses on **customer segmentation and personalization** using the public **Credit Card Customers (CC GENERAL)** dataset.

---

## Architecture (summary)

```text
User → Frontend → FastAPI (/query)
  → Query Manager → Context Builder → Planner
  → Execution Graph → Scheduler → Analytical Nodes
  → Decision Engine (rule-based explanations) → Response Composer → JSON
```

- **API** validates/forwards only — no analytics in route handlers (except legacy `/run-workflow` fallback)
- **Planner** decides *what* to run; **Scheduler** decides *when*; **Nodes** decide *how*
- **Decision Engine** produces lightweight explanations (SHAP/LIME deferred)

See [`Ai files/Static/architecture_short.md`](Ai%20files/Static/architecture_short.md) for the full module map.

---

## Dataset

| Item | Detail |
|------|--------|
| **Name** | Credit Card Customers (CC GENERAL) |
| **Source** | [UCI ML Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| **Local path** | `backend/data/raw/CC GENERAL.csv` |
| **Notes** | Customer-level behavioural features (8950 rows × 18 columns in the full dataset). CSV files are **gitignored** — you must download or bootstrap locally. |

### Bootstrap synthetic demo data (optional)

If you do not have the full UCI file yet:

```powershell
python scripts/bootstrap_sample_data.py
```

This writes a schema-compatible synthetic CSV for local demos and tests.

---

## Setup

### Prerequisites

- Python 3.11+
- Git

### Install

```powershell
git clone https://github.com/ak-1344/Aster---Distributed-Thinking.git
cd Aster---Distributed-Thinking

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# Place real CC GENERAL.csv OR run bootstrap script
python scripts/bootstrap_sample_data.py
```

### Run backend + frontend

From the **repository root**:

```powershell
python backend/main.py
```

Open **http://127.0.0.1:8000/** in a browser. The page POSTs to `/query` and renders segments, recommendations, explanations, and statistics.

### Run tests

```powershell
python -m unittest discover -s backend/tests -p "test_*.py" -v
```

---

## Example queries (for judges)

| Query | Expected behaviour |
|-------|-------------------|
| `segment customers into 3 clusters` | Full segmentation workflow + recommendations + explanations |
| `segment customers into 4 clusters` | Same with 4 clusters (parsed from query) |
| `show descriptive statistics for the dataset` | Analytics + EDA only (no segmentation) |

**API example:**

```powershell
curl -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"query\": \"segment customers into 3 clusters\"}"
```

Legacy bypass (direct nodes, non-canonical): `POST /run-workflow`

---

## Implemented vs out of scope (hackathon)

| Implemented | Out of scope (this submission) |
|-------------|--------------------------------|
| `POST /query` canonical pipeline | `POST /upload`, `GET /status` |
| Rule-based segment explanations | SHAP / LIME |
| Minimal HTML frontend wired to `/query` | Next.js / Tailwind app |
| KMeans segmentation + rule recommendations | LLM planner (Gemini) |
| Execution graph + scheduler | Auth, cache, decision memory |

---

## AI tools disclosure (hackathon requirement)

The following AI assistants/tools were used during development:

- **Cursor IDE** (Composer / agent mode) — code generation, refactoring, test authoring, documentation drafts, commit planning
- **GitHub Copilot-style inline suggestions** (via Cursor) — boilerplate and docstrings

All architectural decisions, module boundaries, and final code were reviewed against `architecture_short.md`, `plan.md`, and project `rules.md`. Human developers retained control of commits, scope, and submission checklist.

---

## License / attribution

- ASTER project scaffold: hackathon team repository
- CC GENERAL dataset: UCI Machine Learning Repository — cite the dataset source above in any academic submission

---

*"Think. Plan. Execute. Explain."*
