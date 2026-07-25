"""Feature engineering helpers for customer-level segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.utils.loader import DEFAULT_DATASET_PATH, load_dataset


DEFAULT_FEATURES_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "customer_features.csv"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two numeric series without infinite values."""

    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_customer_features(
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Build customer-level behavioural features and persist them to CSV."""

    source_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    destination_path = Path(output_path) if output_path is not None else DEFAULT_FEATURES_OUTPUT_PATH

    dataframe = load_dataset(source_path)
    numeric_columns = dataframe.select_dtypes(include="number").columns
    for column in numeric_columns:
        dataframe[column] = dataframe[column].fillna(dataframe[column].median())

    features = pd.DataFrame({"CUST_ID": dataframe["CUST_ID"]})
    features["TENURE"] = dataframe["TENURE"]
    features["monthly_spend"] = _safe_divide(dataframe["PURCHASES"], dataframe["TENURE"])
    features["avg_purchase_ticket"] = _safe_divide(dataframe["PURCHASES"], dataframe["PURCHASES_TRX"])
    features["transaction_frequency_per_month"] = _safe_divide(dataframe["PURCHASES_TRX"], dataframe["TENURE"])
    features["cash_advance_ratio"] = _safe_divide(
        dataframe["CASH_ADVANCE"],
        dataframe["CASH_ADVANCE"] + dataframe["PURCHASES"],
    )
    features["installment_purchase_share"] = _safe_divide(
        dataframe["INSTALLMENTS_PURCHASES"], dataframe["PURCHASES"]
    )
    features["oneoff_purchase_share"] = _safe_divide(dataframe["ONEOFF_PURCHASES"], dataframe["PURCHASES"])
    features["credit_utilization_ratio"] = _safe_divide(dataframe["BALANCE"], dataframe["CREDIT_LIMIT"])
    features["payment_to_minimum_ratio"] = _safe_divide(dataframe["PAYMENTS"], dataframe["MINIMUM_PAYMENTS"])
    features["full_payment_ratio"] = dataframe["PRC_FULL_PAYMENT"]
    features["cash_advance_intensity"] = _safe_divide(dataframe["CASH_ADVANCE_TRX"], dataframe["TENURE"])
    features["credit_headroom"] = dataframe["CREDIT_LIMIT"] - dataframe["BALANCE"]

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination_path, index=False)
    return features, destination_path