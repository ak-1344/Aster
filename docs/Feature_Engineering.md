# Feature Engineering

## Output Artifact

- File: `backend/data/processed/customer_features.csv`
- Shape: `8950 rows x 13 columns`
- Grain: one row per customer

## Engineered Feature Columns

- `CUST_ID`
- `TENURE`
- `monthly_spend`
- `avg_purchase_ticket`
- `transaction_frequency_per_month`
- `cash_advance_ratio`
- `installment_purchase_share`
- `oneoff_purchase_share`
- `credit_utilization_ratio`
- `payment_to_minimum_ratio`
- `full_payment_ratio`
- `cash_advance_intensity`
- `credit_headroom`

## Engineering Notes

- Numeric missing values are median-imputed before feature calculations.
- Ratio calculations use safe division to avoid infinity and divide-by-zero artifacts.
- Features are designed from customer-level input columns because the selected dataset does not provide transaction timestamps or merchant/category-level details.