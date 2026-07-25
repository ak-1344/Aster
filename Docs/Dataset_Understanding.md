# Dataset Understanding

## Selected Dataset

- File: `backend/data/raw/CC GENERAL.csv`
- Shape: `8950 rows x 18 columns`
- Level of analysis: customer-level, not transaction-level

## Identifiers

- Customer identifier: `CUST_ID`
- Transaction identifier: not present in the selected dataset

## Missing Values

- `CREDIT_LIMIT`: 1 missing value
- `MINIMUM_PAYMENTS`: 313 missing values
- All other columns: 0 missing values

## Datatypes

- `CUST_ID`: string
- `BALANCE`, `BALANCE_FREQUENCY`, `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `CASH_ADVANCE`, `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`, `CREDIT_LIMIT`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT`: floating-point numeric columns
- `CASH_ADVANCE_TRX`, `PURCHASES_TRX`, `TENURE`: integer columns

## Notes

- The dataset already represents one row per customer, so Phase 1 focuses on schema understanding rather than transaction aggregation.
- The immediate analytical follow-up for later phases is feature engineering from the existing customer-level behavioural fields.