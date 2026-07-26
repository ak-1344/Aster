# ASTER Setup & Installation Guide

This document provides the canonical steps to install, configure, run, and troubleshoot the ASTER platform locally.

## Prerequisites
*   **Python:** version 3.11 or higher.
*   **Git:** to clone the repository.
*   **Google Gemini API Key:** (Optional) required for LLM-driven planning and reasoning phases. Without it, ASTER uses deterministic rule-based fallback routing.

---

## 1. Clone and Install Dependencies

First, clone the repository and navigate into it:

```bash
git clone https://github.com/ak-1344/Aster---Distributed-Thinking.git
cd Aster---Distributed-Thinking
```

Set up a virtual environment to isolate the project's dependencies:

```bash
# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required Python packages from the backend manifest:

```bash
pip install -r backend/requirements.txt
```

---

## 2. Dataset Bootstrap

ASTER uses the **Credit Card Customers (CC GENERAL)** dataset for its segmentation workflows. Because raw dataset files are ignored by git (`.gitignore`), you must either provide the real dataset or bootstrap a synthetic one.

**Option A: Fast Local Bootstrap (Recommended for testing)**
We provide a script to generate a schema-compatible synthetic CSV for local demos and tests:

```bash
python scripts/bootstrap_sample_data.py
```
*(This writes a synthetic `CC GENERAL.csv` directly into `backend/data/raw/`)*

**Option B: Manual Placement (For real analytics)**
Download `CC GENERAL.csv` from [Kaggle — Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) and place it at:
`backend/data/raw/CC GENERAL.csv`

---

## 3. Environment Configuration (`.env`)

ASTER reads configuration variables directly from your environment or a `.env` file at the root of the repository.

Copy the example file to create your own configuration:
```bash
cp .env.example .env
```

### Required / Recommended Variables

*   **`GEMINI_API_KEY`**: Your Google Gemini API key. If omitted or unreachable, ASTER seamlessly falls back to deterministic rule-based planning.

### Explainability Configuration (`EXPLAINABILITY_MODE`)

**Yes, explainability mode is controlled directly by environment variables!**
ASTER reads `EXPLAINABILITY_MODE` from your `.env` file or shell environment (`os.environ.get("EXPLAINABILITY_MODE", "shap")`).

*   **`EXPLAINABILITY_MODE`**: Controls the surrogate-model explainability engine:
    *   `shap` (default): Fits a Random Forest surrogate model and uses SHAP (`TreeExplainer`) for feature attribution per cluster.
    *   `lime`: Fits a Random Forest surrogate model and uses LIME (`LimeTabularExplainer`) for feature attribution per cluster.
    *   `rule_based`: Bypasses surrogate models entirely and uses fast centroid-distance calculation (this is also the automatic fallback if SHAP or LIME times out or fails).
*   **`GEMINI_TIMEOUT_SECONDS`**: Sets the maximum wait time (in seconds) for LLM responses (default: `10`).
*   **`GEMINI_MODEL`**: Specifies the exact Gemini model to use for planning (default: `gemini-2.5-flash`).

---

## 4. Running the Application

ASTER serves both the backend REST/WebSocket API and the static frontend from a single process.

From the **repository root**, run:

```bash
python backend/main.py
```

*   **Main Chat Interface:** Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
*   **Telemetry Dashboard UI:** Open [http://127.0.0.1:8000/dashboard-ui](http://127.0.0.1:8000/dashboard-ui)
*   **Interactive API Docs (Swagger):** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 5. Troubleshooting: Common Errors

### Error: `[Errno 98] error while attempting to bind on address ('127.0.0.1', 8000): address already in use`

**Cause:** Another process (such as a previously started Uvicorn server or background process) is already listening on port `8000`.

**Resolution:**

**Option 1 (Recommended):** Kill the existing process bound to port 8000.
```bash
# On Linux / macOS:
fuser -k 8000/tcp

# Or find PID and kill manually:
lsof -ti:8000 | xargs kill -9

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

Then re-run:
```bash
python backend/main.py
```

**Option 2:** Access the server that is already running at `http://127.0.0.1:8000/`.

---

## 6. Running the Test Suite

ASTER includes a unit and integration test suite ensuring all modules, dependencies, and fallback paths remain stable.

To run the full suite:

```bash
python -m unittest discover -s backend/tests -p "test_*.py" -v
```

*Note: The test suite runs cleanly with or without a `GEMINI_API_KEY` set.*

