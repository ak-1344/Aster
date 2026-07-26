"""Data preprocessing, feature engineering, K-Means clustering, and product recommendations.

Standalone tool for credit-card customer persona segmentation. This module is
independent of the agentic node pipeline and can be used for demos, notebooks,
or direct retrieval of persona-aligned customer lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from backend.utils.loader import DEFAULT_DATASET_PATH, load_dataset

REQUIRED_COLUMNS = [
    "CUST_ID",
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

CLUSTER_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_UTILIZATION",
    "PAYMENT_RATIO",
    "PURCHASES_FREQUENCY",
]

PERSONA_CATALOG: dict[int, dict[str, Any]] = {
    0: {
        "persona": "The Transactors",
        "profile": "High Purchases, High Full Payments",
        "product": "High-End Premium Travel Rewards Card",
        "sort_key": "PURCHASES",
        "sort_ascending": False,
    },
    1: {
        "persona": "The Revolvers",
        "profile": "High Balance, Low Payment Ratio",
        "product": "Balance Transfer Card / Debt Consolidation Loan",
        "sort_key": "BALANCE",
        "sort_ascending": False,
    },
    2: {
        "persona": "Cash-Advance Reliant",
        "profile": "High Cash Advance, Low Purchases",
        "product": "Low-fee Personal Loan / Overdraft Protection",
        "sort_key": "CASH_ADVANCE",
        "sort_ascending": False,
    },
    3: {
        "persona": "Dormant / Low Activity",
        "profile": "Low Balance Frequency, Low Purchases",
        "product": "Zero-fee Basic Checking / First-purchase Cashback Offer",
        "sort_key": "BALANCE_FREQUENCY",
        "sort_ascending": True,
    },
}

_PERSONA_NAME_TO_CLUSTER = {
    details["persona"].lower(): cluster_id for cluster_id, details in PERSONA_CATALOG.items()
}

# Aliases for convenience lookups.
_PERSONA_ALIASES = {
    "transactors": 0,
    "the transactors": 0,
    "revolvers": 1,
    "the revolvers": 1,
    "cash-advance reliant": 2,
    "cash advance reliant": 2,
    "dormant": 3,
    "dormant / low activity": 3,
    "low activity": 3,
}


def _resolve_persona_cluster(persona_name: str) -> int:
    """Map a persona name (or alias) to its cluster id."""

    key = persona_name.strip().lower()
    if key in _PERSONA_ALIASES:
        return _PERSONA_ALIASES[key]
    if key in _PERSONA_NAME_TO_CLUSTER:
        return _PERSONA_NAME_TO_CLUSTER[key]
    raise ValueError(
        f"Unknown persona '{persona_name}'. "
        f"Expected one of: {[details['persona'] for details in PERSONA_CATALOG.values()]}"
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values for MINIMUM_PAYMENTS and CREDIT_LIMIT with medians."""

    frame = df.copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    for column in ("MINIMUM_PAYMENTS", "CREDIT_LIMIT"):
        median_value = float(frame[column].median(skipna=True))
        frame[column] = frame[column].fillna(median_value)

    return frame


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add CREDIT_UTILIZATION, PAYMENT_RATIO, and INSTALLMENT_RATIO features."""

    frame = df.copy()

    credit_limit = frame["CREDIT_LIMIT"].replace(0, np.nan)
    utilization = frame["BALANCE"] / credit_limit
    frame["CREDIT_UTILIZATION"] = utilization.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=1.0)

    payment_ratio = frame["PAYMENTS"] / frame["MINIMUM_PAYMENTS"].replace(0, np.nan)
    frame["PAYMENT_RATIO"] = payment_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    frame["INSTALLMENT_RATIO"] = frame["INSTALLMENTS_PURCHASES"] / (frame["PURCHASES"] + 1e-5)

    return frame


def fit_kmeans_personas(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Scale selected features, fit K-Means (k=4), and attach persona labels."""

    frame = df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame[CLUSTER_FEATURES])

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = model.fit_predict(scaled)

    frame["cluster"] = cluster_labels.astype(int)
    frame["persona"] = frame["cluster"].map(lambda c: PERSONA_CATALOG[int(c)]["persona"])
    frame["recommended_product"] = frame["cluster"].map(
        lambda c: PERSONA_CATALOG[int(c)]["product"]
    )
    frame["persona_profile"] = frame["cluster"].map(lambda c: PERSONA_CATALOG[int(c)]["profile"])

    return frame, model, scaler


def preprocess_and_cluster(
    dataset_path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    n_clusters: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run full ingest → preprocess → feature engineering → K-Means pipeline."""

    if df is None:
        source = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
        frame = load_dataset(source)
        source_path = str(source)
    else:
        frame = df.copy()
        source_path = None

    prepared = engineer_features(preprocess(frame))
    clustered, model, scaler = fit_kmeans_personas(
        prepared,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    persona_summary = (
        clustered.groupby(["cluster", "persona", "recommended_product"], as_index=False)
        .size()
        .rename(columns={"size": "customer_count"})
        .to_dict(orient="records")
    )

    return {
        "dataframe": clustered,
        "model": model,
        "scaler": scaler,
        "feature_columns": list(CLUSTER_FEATURES),
        "persona_catalog": PERSONA_CATALOG,
        "persona_summary": persona_summary,
        "dataset_path": source_path,
        "n_clusters": n_clusters,
        "random_state": random_state,
    }


def get_top_customers_for_persona(
    persona_name: str,
    df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top N customers for a persona, sorted by key relevance."""

    if "persona" not in df.columns and "cluster" not in df.columns:
        raise ValueError("DataFrame must include persona/cluster columns from preprocess_and_cluster().")

    cluster_id = _resolve_persona_cluster(persona_name)
    catalog = PERSONA_CATALOG[cluster_id]
    sort_key = catalog["sort_key"]
    ascending = bool(catalog["sort_ascending"])

    if "cluster" in df.columns:
        subset = df[df["cluster"] == cluster_id].copy()
    else:
        subset = df[df["persona"].str.lower() == catalog["persona"].lower()].copy()

    if sort_key not in subset.columns:
        raise ValueError(f"Missing sort key column '{sort_key}' for persona '{catalog['persona']}'.")

    return subset.sort_values(by=sort_key, ascending=ascending).head(top_n).reset_index(drop=True)


def get_customer_insight(cust_id: str, df: pd.DataFrame) -> dict[str, Any]:
    """Return a detailed breakdown for a single CUST_ID."""

    if "CUST_ID" not in df.columns:
        raise ValueError("DataFrame must include CUST_ID.")

    matches = df[df["CUST_ID"].astype(str) == str(cust_id)]
    if matches.empty:
        raise KeyError(f"Customer '{cust_id}' not found.")

    row = matches.iloc[0]
    cluster = int(row["cluster"]) if "cluster" in matches.columns and pd.notna(row.get("cluster")) else None
    catalog = PERSONA_CATALOG.get(cluster, {}) if cluster is not None else {}

    return {
        "cust_id": str(row["CUST_ID"]),
        "cluster": cluster,
        "persona": row.get("persona", catalog.get("persona")),
        "persona_profile": row.get("persona_profile", catalog.get("profile")),
        "recommended_product": row.get("recommended_product", catalog.get("product")),
        "metrics": {
            "BALANCE": float(row["BALANCE"]) if "BALANCE" in row else None,
            "BALANCE_FREQUENCY": float(row["BALANCE_FREQUENCY"]) if "BALANCE_FREQUENCY" in row else None,
            "PURCHASES": float(row["PURCHASES"]) if "PURCHASES" in row else None,
            "CASH_ADVANCE": float(row["CASH_ADVANCE"]) if "CASH_ADVANCE" in row else None,
            "CREDIT_LIMIT": float(row["CREDIT_LIMIT"]) if "CREDIT_LIMIT" in row else None,
            "PAYMENTS": float(row["PAYMENTS"]) if "PAYMENTS" in row else None,
            "MINIMUM_PAYMENTS": float(row["MINIMUM_PAYMENTS"]) if "MINIMUM_PAYMENTS" in row else None,
            "PRC_FULL_PAYMENT": float(row["PRC_FULL_PAYMENT"]) if "PRC_FULL_PAYMENT" in row else None,
            "PURCHASES_FREQUENCY": float(row["PURCHASES_FREQUENCY"]) if "PURCHASES_FREQUENCY" in row else None,
            "CREDIT_UTILIZATION": float(row["CREDIT_UTILIZATION"]) if "CREDIT_UTILIZATION" in row else None,
            "PAYMENT_RATIO": float(row["PAYMENT_RATIO"]) if "PAYMENT_RATIO" in row else None,
            "INSTALLMENT_RATIO": float(row["INSTALLMENT_RATIO"]) if "INSTALLMENT_RATIO" in row else None,
        },
    }


def run_segmentation_pipeline(
    dataset_path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    n_clusters: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Public alias for the full preprocessing + K-Means persona pipeline."""

    return preprocess_and_cluster(
        dataset_path=dataset_path,
        df=df,
        n_clusters=n_clusters,
        random_state=random_state,
    )


def run_segmentation_tool(
    dataset_path: str | Path | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Convenience entrypoint: cluster customers and preview top rows per persona."""

    result = run_segmentation_pipeline(dataset_path=dataset_path)
    clustered: pd.DataFrame = result["dataframe"]

    top_by_persona = {
        details["persona"]: get_top_customers_for_persona(details["persona"], clustered, top_n=top_n)
        .loc[:, ["CUST_ID", "persona", "recommended_product", details["sort_key"]]]
        .to_dict(orient="records")
        for details in PERSONA_CATALOG.values()
    }

    return {
        "persona_summary": result["persona_summary"],
        "top_customers_by_persona": top_by_persona,
        "customer_count": int(len(clustered)),
        "feature_columns": result["feature_columns"],
        "dataframe": clustered,
        "model": result["model"],
        "scaler": result["scaler"],
    }
