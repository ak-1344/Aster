# ASTER — Customer Segmentation & Personalization Agent for Retail Banking

> **Agentic Segmentation Through Execution & Reasoning (ASTER)** transforms natural-language business questions into planned analytical workflows, executes them through modular nodes, and returns explainable segmentation insights and recommendations.

ASTER is an intelligent analytics assistant for retail banking teams. Instead of running a fixed pipeline for every question or forcing analysts to write code, ASTER understands what you want, dynamically plans the right analytical steps (e.g. data cleaning → clustering → recommendation generation), executes them, and translates the mathematical results back into plain-English business insights.

*"Think. Plan. Execute. Explain."*

---

## Documentation Index

The complete, verified documentation for ASTER is located in the `docs/` directory:

1. 🏛️ **[Architecture Overview](docs/ARCHITECTURE.md)**: A deep dive into how ASTER separates planning, scheduling, computation, and explainability. Includes the request flow diagram and module responsibilities.
2. ✨ **[Features Guide](docs/FEATURES.md)**: A user-centric breakdown of what ASTER can actually do (Automated EDA, Segmentation, Explainability, etc.) and what is driven by AI versus what remains deterministic.
3. 🔄 **[User Workflow](docs/USER_WORKFLOW.md)**: A step-by-step walkthrough of what happens under the hood when a user asks a question, using realistic examples of increasing complexity (including cache hits).
4. ⚙️ **[Setup & Installation](docs/SETUP.md)**: Instructions for installing dependencies, configuring environment variables (like `GEMINI_API_KEY`), bootstrapping the dataset, and running the test suite.

---

## Quick Start

If you want to jump right in, follow these steps (or see [Setup](docs/SETUP.md) for more details):

```bash
# Clone the repository
git clone https://github.com/ak-1344/Aster---Distributed-Thinking.git
cd Aster---Distributed-Thinking

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # (On Windows: .\.venv\Scripts\Activate.ps1)

# Install dependencies
pip install -r backend/requirements.txt

# Bootstrap the synthetic dataset (if you don't have the real one)
python scripts/bootstrap_sample_data.py

# Configure environment variables (add your GEMINI_API_KEY)
cp .env.example .env

# Run the backend and frontend
python backend/main.py
```

Navigate to `http://127.0.0.1:8000/` in your browser to use the interface!

---

## Example Queries (for judges/testing)

Try typing these into the frontend chat interface to exercise different parts of the system:

| Query | Expected Behaviour |
|-------|--------------------|
| `segment customers into 3 clusters` | Full segmentation workflow + recommendations + explanations. |
| `show descriptive statistics for the dataset` | Analytics + EDA only (no clustering). |
| `Why was customer 1004 placed in their current segment?` | Explanation-only follow-up. Skips analytics and runs the surrogate explainability model. |
| `segment customers into 3 clusters` *(run again)* | Decision Memory Cache Hit. Near-instant response bypassing the pipeline. |

---

## Dataset Attribution

| Item | Detail |
|------|--------|
| **Name** | Credit Card Customers (CC GENERAL) |
| **Source** | [UCI ML Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| **Notes** | Customer-level behavioural features (8950 rows × 18 columns in the full dataset). CSV files are **gitignored** — you must download or bootstrap locally. |

---

## AI Tools Disclosure (Hackathon Requirement)

The following AI assistants/tools were used during development:

*   **Cursor IDE** (Composer / agent mode) — code generation, refactoring, test authoring, documentation drafts, commit planning.
*   **GitHub Copilot-style inline suggestions** (via Cursor) — boilerplate and docstrings.

All architectural decisions, module boundaries, and final code were reviewed against `architecture_short.md`, `plan.md`, and project `rules.md`. Human developers retained control of commits, scope, and submission checklist.

---

## License / Attribution

*   ASTER project scaffold: hackathon team repository
*   CC GENERAL dataset: UCI Machine Learning Repository — cite the dataset source above in any academic submission
