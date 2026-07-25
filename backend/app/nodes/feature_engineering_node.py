"""Feature engineering node for ASTER."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.utils.feature_engineering import build_customer_features


def generate_features(
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Generate customer-level features for downstream segmentation."""

    return build_customer_features(dataset_path=dataset_path, output_path=output_path)