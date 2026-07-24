Alright, let's map this out properly — assuming near-zero AI/ML background, here's how I'd actually run this.

## Step-by-step procedure

**Day 0 (before the clock starts)**
1. Set up the GitHub repo (public, empty commit to mark start time), Python env, and pick your dataset now — you're not allowed to use the actual problem statement time for dataset hunting if you can prep in advance. Look at Kaggle for "bank customer segmentation" or "credit card customer segmentation" datasets — these exist ready-made with balance, transaction frequency, tenure, product holdings columns.
2. Skim 2-3 articles on RFM analysis (Recency, Frequency, Monetary) — this is the classic non-ML-heavy way banks segment customers, and it'll make you sound domain-fluent without needing deep ML.

**Hours 0-6: Data + EDA**
3. Load the dataset, clean it (nulls, duplicates, types).
4. Write a standalone EDA script/notebook: distributions, correlations, missing data report. This becomes your "EDA tool" logic later — just write it as a plain function first, wire it into the agent after.

**Hours 6-14: Segmentation core**
5. Feature engineering: derive RFM-style features (transaction frequency, average balance, product count, tenure).
6. Run K-means or hierarchical clustering (scikit-learn — 5 lines of code, don't overthink this). Pick cluster count via elbow method or silhouette score.
7. Name and describe each cluster in plain English ("high-balance low-activity", "young high-frequency low-balance", etc.) — this becomes your personas.

**Hours 14-22: Explainability + recommendations**
8. Add SHAP or simple feature-importance-per-cluster to explain *why* a customer landed in a segment (this satisfies the explainability requirement without needing deep ML knowledge — SHAP has a near copy-paste API for tree models).
9. Write rule-based recommendation logic per segment (e.g. "low product count + high balance → recommend investment products"). Rule-based is explicitly allowed and is easier to defend in Q&A than a black-box model.

**Hours 22-32: Wrap it in an agent**
10. This is the part most teams will get wrong or skip. Build a thin orchestrator (see architecture below) that takes a natural-language query, decides which of your already-built tools to call, and returns a synthesized answer. This is the single highest-leverage thing you can do since it's explicitly graded and most teams will bolt on a chatbot as an afterthought.

**Hours 32-40: Front-end + polish**
11. A simple Streamlit or minimal React+FastAPI chat UI. Streamlit is faster given your timeline and low ML background.
12. Visualizations: cluster scatter plots (PCA/t-SNE reduced to 2D), bar charts of segment sizes, feature importance charts.

**Hours 40-48: Deliverables**
13. README (dataset citation, setup, architecture, usage).
14. 2-slide deck (I can help you build this).
15. 2-minute demo video script — plan this in advance, don't wing it at hour 47.
16. Final commit cleanup — make sure commit history actually shows incremental work, not one dump.

## Tech stack

Given your background (you're comfortable with Python, prefer minimal glue code, and have said ML feels oversaturated to you) — lean toward the *lightest* stack that satisfies "agentic," not the trendiest one:

- **Orchestration**: LangGraph (not raw LangChain) — it's built exactly for "decide which tool to call based on state," which is literally what the problem statement demands. A plain LangChain agent executor works too and is less code, but LangGraph gives you a visual, defensible graph structure for the round-2 explanation.
- **LLM**: Groq's free tier (Llama 3.1/3.3) or Gemini free tier — fast and free, good enough for query-parsing/intent-extraction, no GPU needed.
- **Data/ML**: pandas, scikit-learn (KMeans, StandardScaler, PCA), SHAP for explainability.
- **Visualization**: matplotlib/plotly (plotly if you want interactive charts in the front-end).
- **Front-end**: Streamlit (fastest path to a working chat UI given your timeline; skip React unless you specifically want the practice).
- **API layer** (optional, only if you want chat+API both): FastAPI, thin wrapper around the same agent.

This is a small, boring stack on purpose — the org explicitly says they don't want production-grade complexity, they want explainable and complete.

## Architecture

I put this in the diagram above — orchestrator agent sits in the middle, decides which of the four tools to invoke per query, doesn't run all of them every time. Concretely:

- **Orchestrator (LangGraph)**: parses intent from the query, routes to a subset of tools, synthesizes the final answer from whatever tools ran.
- **EDA tool**: runs only when the query needs raw data understanding ("what does the data look like").
- **Feature engineering tool**: builds/derives the features needed for whatever segmentation the query implies.
- **Segmentation tool**: clusters or rule-assigns customers, flags edge cases (customers near a segment boundary).
- **Explainability tool**: answers "why is this customer in segment X" using SHAP or feature-contribution logic.

The recommendation logic (cross-sell/up-sell rules) can live as a fifth callable tool or as a post-processing step after segmentation — either is fine, just document which you chose.

## User story

> Priya, an analyst at a retail bank, opens the tool and types: *"Which customers could be upgraded from regular to priority status, and why?"*
>
> The agent parses this — realizes it doesn't need to re-run full EDA, just needs segmentation + explainability. It pulls the customer base, checks which "regular" customers are near the priority-tier boundary on balance/transaction-frequency, and returns: a short list of borderline customers, the specific features pushing them close (e.g. "transaction frequency in top 15%, balance just below threshold"), and a suggested nudge ("offer a savings-to-investment cross-sell to accelerate balance growth").
>
> Priya then asks a follow-up: *"Compare average transaction size between this group and the existing priority segment."* The agent doesn't restart — it reuses the segments already computed and just runs a comparison, returning a chart. She never touched a notebook.

This story is worth keeping in your README/demo script almost verbatim — it directly demonstrates "adaptive execution," which is the thing the judges said they care about most.

## What would make you stand out

Most teams will build: cluster once, hardcode a few plots, and put a chatbot skin on top that always runs the full pipeline regardless of the question. To beat that:

1. **Actually make routing visible.** Show, in the UI, which tools got invoked for a given query (a small "agent trace" panel). This proves adaptive execution instead of just claiming it — judges can *see* that a different query triggered a different path.
2. **Handle edge cases explicitly.** The doc calls out "identify regular customers who could become priority customers" — build a genuine boundary-detection feature (distance-to-centroid or distance-to-threshold), not just static cluster labels. This is a small addition that most teams will skip.
3. **Segment stability check.** Run clustering with 2-3 different k values or a bootstrap resample and show how stable the segments are — a one-paragraph "how confident are we in these segments" section separates you from teams that just picked k=4 and moved on.
4. **Natural-language explainability, not just SHAP plots.** Take the SHAP output and have the LLM turn it into a sentence ("this customer is flagged priority-eligible mainly due to rising transaction frequency over the last quarter, not raw balance"). Judges are reviewers, not data scientists — readable explanations score better than a chart with feature bars.
5. **Tight commit story.** Since round 2 explicitly checks whether GitHub history reflects each member's contribution, structure commits so that's genuinely visible (don't squash everything the night before).
