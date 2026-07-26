# ASTER Setup & Installation Guide

This document provides the canonical steps to install, configure, and run the ASTER platform locally.

## Prerequisites
*   **Python:** version 3.11 or higher.
*   **Git:** to clone the repository.
*   **Google Gemini API Key:** required for the LLM-driven planning and reasoning phases.

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
We provide a script to generate a schema-compatible synthetic CSV for local demos and tests.

```bash
python scripts/bootstrap_sample_data.py
```
*(This writes a synthetic `CC GENERAL.csv` directly into `backend/data/raw/`)*

**Option B: Manual Placement (For real analytics)**
Download the `CC GENERAL.csv` from [Kaggle — Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) and place it exactly at:
`backend/data/raw/CC GENERAL.csv`

---

## 3. Environment Configuration

ASTER reads configuration from environment variables. You can set these in your shell or place them in a `.env` file at the root of the repository. 

Copy the example file to create your own configuration:
```bash
cp .env.example .env
```

### Required Variables

*   **`GEMINI_API_KEY`**: Your Google Gemini API key. Without this, the pipeline will fall back to rule-based execution for all queries.

### Optional Configuration / Toggles

*   **`EXPLAINABILITY_MODE`**: Controls the surrogate-model explainability engine.
    *   `shap` (default): Uses SHAP `TreeExplainer` for feature attribution.
    *   `lime`: Uses LIME `TabularExplainer` for feature attribution.
    *   `rule_based`: Bypasses surrogate models entirely and uses a fast, centroid-distance rule calculation (this is also the automatic fallback if SHAP/LIME times out or fails).
*   **`GEMINI_TIMEOUT_SECONDS`**: Sets the maximum time (in seconds) the system will wait for a structured LLM response before triggering deterministic fallbacks. 
    *   *Default: `10`*
*   **`GEMINI_MODEL`**: Specifies the exact Gemini model to use for planning.
    *   *Default: `gemini-2.5-flash`*

---

## 4. Running the Application

ASTER serves both the backend API and the static frontend from a single FastAPI process.

From the **repository root**, run:

```bash
python backend/main.py
```

*   **Dashboard:** Open a browser and navigate to `http://127.0.0.1:8000/` to use the chat interface.
*   **API Docs:** Navigate to `http://127.0.0.1:8000/docs` to view the auto-generated Swagger UI.

---

## 5. Running the Test Suite

ASTER includes a comprehensive smoke and unit test suite ensuring all modules, dependencies, and execution paths remain stable.

To run the full suite, execute:

```bash
python -m unittest discover -s backend/tests -p "test_*.py" -v
```

*Note: The test suite includes LLM integration smoke tests. If `GEMINI_API_KEY` is not set, or the API is unreachable, the system will seamlessly exercise the deterministic fallback paths instead, and the tests should still pass.*
