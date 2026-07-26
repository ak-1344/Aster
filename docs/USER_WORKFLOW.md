# ASTER User Workflow

This document walks through the ASTER experience from a user's perspective, showing how natural-language queries are processed by the system and returned as actionable insights. 

## The Frontend Flow (`frontend/index.html`)
The user experience begins in the ASTER frontend dashboard.
1.  **Input:** The user types a plain-English request into the chat-like query box (e.g., "Group my top spenders").
2.  **Submission:** When the user clicks "Run Query," the frontend sends a `POST /query` request to the ASTER API.
3.  **Processing:** A loading indicator appears while the backend builds and executes the analytical pipeline.
4.  **Results:** Once the API responds, the frontend dynamically renders the JSON payload. The user sees:
    *   The execution workflow that was planned and run.
    *   Aggregated segment summaries and their defining features.
    *   Personalised product recommendations for each group.
    *   Business-focused explanations for why the groupings were made.
    *   Interactive visualisations and descriptive statistics.

---

## Example 1: Simple Descriptive Query (EDA-Only)

**The Query:** *"Show me the descriptive statistics for our customer dataset."*

**Internal Pipeline:**
1.  **Query Manager & Context Builder:** ASTER recognises this as a simple descriptive request. It notes that no clustering or recommendations are required.
2.  **Planner:** The LLM (or rule-based fallback) plans a short workflow containing only the `analytics` and `eda` nodes.
3.  **Scheduler:** The scheduler runs the `analytics` node (to calculate dataset-wide stats like mean and standard deviation), followed by the `eda` node (to generate distributional summaries).
4.  **Decision Engine:** Since there is no segmentation, the explainability engine is skipped.

**What the User Sees:**
The dashboard updates with a high-level summary of the dataset. The user sees tables outlining missing values, data distributions, and overall customer base statistics. There are no segments or recommendations shown.

---

## Example 2: Full Segmentation Workflow

**The Query:** *"Segment customers into 3 clusters based on their spending behaviour."*

**Internal Pipeline:**
1.  **Context Builder:** Extracts the core intent (segmentation) and the specific parameter (3 clusters). 
2.  **Planner:** Determines a full workflow is needed: `analytics` -> `eda` -> `feature_engineering` -> `segmentation` -> `recommendation` -> `evaluation` -> `visualization`.
3.  **Scheduler & Nodes:** The pipeline runs in strict order. `feature_engineering` transforms the raw transactions into behavioural metrics (like `monthly_spend`). The `segmentation` node uses Gemini to select the best algorithm (e.g., KMeans with `n_clusters=3`) and runs the clustering. The `recommendation` node applies business rules and uses the LLM to write natural-sounding suggestions for each cluster.
4.  **Decision Engine:** The system fits a surrogate model on the newly created clusters and runs SHAP (or LIME) to calculate exactly which features drove the segmentation.

**What the User Sees:**
The dashboard presents a comprehensive report:
*   **Workflow Metadata:** Shows the full chain of nodes executed and notes that "KMeans" was the algorithm chosen by the LLM.
*   **Segments:** Detailed breakdowns of the 3 clusters, including size and key defining traits.
*   **Recommendations:** Business-friendly text like, "Segment 0 users have high credit utilisation; recommend a balance transfer card."
*   **Explanations:** Clear statements like, "Assigned to Segment 0 primarily due to high credit utilisation and low payment-to-minimum ratio."
*   **Visualisations:** Interactive scatter plots showing the customer groupings.

---

## Example 3: Explanation-Only Follow-Up

**The Query:** *"Why was customer 1004 placed in their current segment?"*

**Internal Pipeline:**
1.  **Context Builder:** Identifies the intent as `explanation_only` and extracts the entity `customer 1004`.
2.  **Planner:** Recognises that no new analytics are needed. The execution graph contains zero analytical nodes.
3.  **Scheduler:** Skips the analytical nodes entirely.
4.  **Decision Engine:** The engine pulls the *existing* clustering output and engineered features from memory. It runs the surrogate model explainer targeting only the index for customer 1004.

**What the User Sees:**
The dashboard bypasses the heavy charts and tables, returning a direct, fast response explaining the specific feature contributions that placed customer 1004 in their assigned segment.

---

## Example 4: Decision Memory (Cache Hit)

**The Query:** *"Segment customers into 3 clusters based on their spending behaviour."* (Run a second time).

**Internal Pipeline:**
1.  **Query Manager:** Before normalising the query and sending it to the Planner, the system checks the SQLite-backed Decision Memory cache for an exact match.
2.  **Cache Hit:** It finds the exact query ("segment customers into 3 clusters based on their spending behaviour") was successfully run previously.
3.  **Bypass:** The entire Planner, Execution Graph, Scheduler, Node Computation, and Decision Engine stack is completely bypassed.
4.  **Response:** The system immediately returns the stored JSON response from the database, appending `cache_hit: true` and the original `cached_created_at` timestamp.

**What the User Sees:**
The response is near-instantaneous (skipping the typical 3-5 second LLM/clustering delay). The frontend renders the exact same comprehensive report as Example 2, but the user will notice a `cache_hit` flag and timestamp in the metadata panel, indicating the results were served from memory.

---

## Example 5: Custom Dataset Upload & Querying

**The Action:** Uploading a custom dataset `new_customers.csv` via the dashboard upload drop zone.

**Internal Pipeline:**
1.  **Frontend:** Transmits the file via `POST /upload`.
2.  **Dataset Manager:** Validates CSV structure, computes missing value imputations, generates `dataset_id` (e.g., `ds_a8b9f1`), writes raw data to `backend/data/raw/ds_a8b9f1.csv`, and runs feature engineering creating `backend/data/processed/ds_a8b9f1_features.csv`.
3.  **Subsequent Queries:** The user passes `dataset_id: "ds_a8b9f1"` in subsequent `/query` calls.

**What the User Sees:**
A green toast notification confirms dataset ingestion with row count, column list, and generated features. The query interface updates to target the newly uploaded dataset.

---

## Example 6: Query with Unsupported Search Filters

**The Query:** *"Segment customers into 3 clusters who live in New York and are over age 30."*

**Internal Pipeline:**
1.  **Context Builder:** Parses the query and extracts requested filters (`city = New York`, `age > 30`).
2.  **Filter Validation:** Compares requested filters against the loaded dataset schema (`CC GENERAL.csv`). Notes that neither `city` nor `age` exists in the schema.
3.  **Surfacing:** Appends `["city", "age"]` to `unsupported_filters` in the context payload while retaining core segmentation intent.
4.  **Response Composer & UI:** Assembles response with `unsupported_filters: ["city", "age"]`.

**What the User Sees:**
An alert banner appears on the dashboard warning: *"The following requested query filters are not supported by the dataset schema and were omitted: city, age."* The segmentation analysis continues on the remaining available behavioral metrics without throwing an error.

