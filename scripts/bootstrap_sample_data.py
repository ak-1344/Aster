#!/usr/bin/env python3
"""Create a synthetic CC GENERAL-compatible CSV when the real dataset is not present."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_sample_data(rows: int = 120, seed: int = 42) -> Path:
	"""Write a minimal customer-level dataset matching the CC GENERAL schema."""

	rng = np.random.default_rng(seed)
	target = Path(__file__).resolve().parents[1] / "backend" / "data" / "raw" / "CC GENERAL.csv"
	target.parent.mkdir(parents=True, exist_ok=True)

	if target.exists():
		print(f"Dataset already exists: {target}")
		return target

	frame = pd.DataFrame(
		{
			"CUST_ID": [f"C{i:04d}" for i in range(rows)],
			"BALANCE": rng.uniform(0, 5000, rows),
			"BALANCE_FREQUENCY": rng.uniform(0, 1, rows),
			"PURCHASES": rng.uniform(0, 3000, rows),
			"ONEOFF_PURCHASES": rng.uniform(0, 1500, rows),
			"INSTALLMENTS_PURCHASES": rng.uniform(0, 1500, rows),
			"CASH_ADVANCE": rng.uniform(0, 1000, rows),
			"PURCHASES_FREQUENCY": rng.uniform(0, 1, rows),
			"ONEOFF_PURCHASES_FREQUENCY": rng.uniform(0, 1, rows),
			"PURCHASES_INSTALLMENTS_FREQUENCY": rng.uniform(0, 1, rows),
			"CASH_ADVANCE_FREQUENCY": rng.uniform(0, 1, rows),
			"CASH_ADVANCE_TRX": rng.integers(0, 10, rows),
			"PURCHASES_TRX": rng.integers(1, 30, rows),
			"CREDIT_LIMIT": rng.uniform(1000, 15000, rows),
			"PAYMENTS": rng.uniform(0, 2000, rows),
			"MINIMUM_PAYMENTS": rng.uniform(0, 500, rows),
			"PRC_FULL_PAYMENT": rng.uniform(0, 1, rows),
			"TENURE": rng.integers(6, 48, rows),
		}
	)
	frame.to_csv(target, index=False)
	print(f"Wrote synthetic demo dataset ({rows} rows) to {target}")
	return target


if __name__ == "__main__":
	bootstrap_sample_data()
