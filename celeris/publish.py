"""Forecast generation, lifecycle enforcement, pricing projection, and submission validation for Celeris."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from celeris.forecast import (
    SeasonalNaiveModel,
    TrailingMeanModel,
    PooledDirectModel,
    IntermittentChallengerModel,
    EnsembleModel,
)


@dataclass
class SubmissionValidationReport:
    """Validation report for the generated submission CSV."""
    is_valid: bool
    total_rows: int
    expected_rows: int
    columns: List[str]
    expected_columns: List[str]
    has_nulls: bool
    null_count: int
    all_non_negative: bool
    all_integers: bool
    keys_match_exactly: bool
    forecast_agreement: bool
    total_units_forecast: int
    total_projected_revenue: float
    notes: List[str]


def compute_horizon(forecast_date: str, cutoff_date: str = "2026-10") -> int:
    """Calculate integer horizon in months between cutoff and forecast month."""
    c_year, c_month = map(int, cutoff_date.split("-"))
    f_year, f_month = map(int, forecast_date.split("-"))
    return (f_year - c_year) * 12 + (f_month - c_month)


def project_sku_price(
    sku_id: str,
    forecast_date: str,
    historical_prices: Dict[str, float],
    sku_master_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
) -> float:
    """Project SKU unit price in USD for forecast_date.

    Carries last reliable base price forward and applies documented January list-price increases.
    """
    base_price = historical_prices.get(sku_id, 0.0)

    # If missing in historical transactions, check predecessor or master
    if base_price <= 0 and sku_master_df is not None and not sku_master_df.empty:
        sku_row = sku_master_df[sku_master_df["SKU_ID"] == sku_id]
        if not sku_row.empty:
            pred_sku = str(sku_row.iloc[0].get("Replaces_SKU", "")).strip()
            if pred_sku and pred_sku in historical_prices:
                base_price = historical_prices[pred_sku]
            else:
                base_price = 100.0  # Default non-zero fallback
        else:
            base_price = 100.0

    # Apply 2% annual price increase on Jan 1, 2027 and Jan 1, 2028
    f_year = int(forecast_date.split("-")[0])
    if f_year == 2027:
        price = base_price * 1.02
    elif f_year >= 2028:
        price = base_price * 1.02 * 1.02
    else:
        price = base_price

    return round(price, 2)


def generate_final_forecast(
    history_df: pd.DataFrame,
    forecast_keys_df: pd.DataFrame,
    sku_master_df: pd.DataFrame,
    winning_model_name: str = "pooled_direct",
    events_df: Optional[pd.DataFrame] = None,
    cutoff_date: str = "2026-10",
) -> pd.DataFrame:
    """Generate final 19-month unconstrained point forecast for each supplied test key.

    Implements:
    - Winning model fitting through final cutoff (2026-10)
    - Full lifecycle bounds checking (zero sales before Launch_Date or after EOL_Date)
    - Predecessor cold-start fallback
    - Price projection and projected revenue
    """
    as_of_history = history_df[history_df["Date"] <= cutoff_date].copy()

    # Fit candidate models on full history
    snaive = SeasonalNaiveModel().fit(as_of_history, cutoff_date)
    tmean = TrailingMeanModel().fit(as_of_history, cutoff_date)
    pooled = PooledDirectModel().fit(as_of_history, cutoff_date)
    intermittent = IntermittentChallengerModel().fit(as_of_history, cutoff_date, pooled_model=pooled)
    ensemble = EnsembleModel().fit(as_of_history, cutoff_date, snaive=snaive, pooled=pooled, tmean=tmean)

    models = {
        "seasonal_naive": snaive,
        "trailing_mean": tmean,
        "pooled_direct": pooled,
        "intermittent_challenger": intermittent,
        "ensemble": ensemble,
    }
    selected_model = models.get(winning_model_name, pooled)

    # SKU master metadata lookup
    sku_meta: Dict[str, dict] = {}
    if not sku_master_df.empty:
        for r in sku_master_df.to_dict("records"):
            sku_meta[str(r["SKU_ID"])] = r

    # Historical prices lookup
    valid_prices = as_of_history.dropna(subset=["Unit_Price_USD"])
    if not valid_prices.empty:
        last_prices = valid_prices.groupby("SKU_ID")["Unit_Price_USD"].last().to_dict()
    else:
        last_prices = {}

    base_price_by_sku: Dict[str, float] = {}
    for sku, meta in sku_meta.items():
        if sku in last_prices and last_prices[sku] > 0:
            base_price_by_sku[sku] = float(last_prices[sku])
        else:
            pred_sku = str(meta.get("Replaces_SKU", "")).strip() if pd.notna(meta.get("Replaces_SKU")) else ""
            if pred_sku and pred_sku in last_prices and last_prices[pred_sku] > 0:
                base_price_by_sku[sku] = float(last_prices[pred_sku])
            else:
                base_price_by_sku[sku] = 100.0

    # Historical series observed
    observed_series = set(zip(as_of_history["SKU_ID"], as_of_history["Region"]))

    # Pre-calculate family averages
    family_averages: Dict[str, int] = {}
    for fam, group in as_of_history.groupby("Product_Family"):
        tail = group["reconstructed_demand"].dropna().tail(12)
        if not tail.empty:
            family_averages[fam] = int(round(max(0.0, float(tail.mean()))))
        else:
            family_averages[fam] = 10

    forecast_records = forecast_keys_df.to_dict("records")
    n_records = len(forecast_records)
    forecast_rows: List[dict] = [None] * n_records  # type: ignore

    # Group record indices by date
    records_by_date: Dict[str, List[int]] = {}
    for idx, row in enumerate(forecast_records):
        d = str(row["Date"]).strip()
        records_by_date.setdefault(d, []).append(idx)

    for d, indices in records_by_date.items():
        h = compute_horizon(d, cutoff_date)
        f_year = int(d.split("-")[0])
        price_multiplier = 1.0
        if f_year == 2027:
            price_multiplier = 1.02
        elif f_year >= 2028:
            price_multiplier = 1.02 * 1.02

        batch_keys: List[Tuple[str, str]] = []
        batch_indices: List[int] = []
        batch_models: List[str] = []

        for idx in indices:
            row = forecast_records[idx]
            sku = str(row["SKU_ID"]).strip()
            reg = str(row["Region"]).strip()

            meta = sku_meta.get(sku, {})
            launch_date = str(meta.get("Launch_Date", "")).strip() if pd.notna(meta.get("Launch_Date")) else ""
            eol_date = str(meta.get("EOL_Date", "")).strip() if pd.notna(meta.get("EOL_Date")) else ""
            replaces_sku = str(meta.get("Replaces_SKU", "")).strip() if pd.notna(meta.get("Replaces_SKU")) else ""

            base_price = base_price_by_sku.get(sku, last_prices.get(sku, 100.0))
            unit_price = round(base_price * price_multiplier, 2)

            # 1. Lifecycle Check: Post-EOL
            if eol_date and d > eol_date:
                forecast_rows[idx] = {
                    "Date": d,
                    "SKU_ID": sku,
                    "Region": reg,
                    "Product_Family": row.get("Product_Family", meta.get("Product_Family", "")),
                    "Business_Unit": row.get("Business_Unit", meta.get("Business_Unit", "")),
                    "SKU_Description": row.get("SKU_Description", meta.get("SKU_Description", "")),
                    "Customer_Name": row.get("Customer_Name", ""),
                    "Sales_Channel": row.get("Sales_Channel", ""),
                    "horizon": h,
                    "Units_Sold_Forecast": 0,
                    "Unit_Price_USD": unit_price,
                    "Projected_Revenue_USD": 0.0,
                    "Model_Or_Fallback": "lifecycle_post_eol",
                    "Lifecycle_Status": "Discontinued",
                }
            # 2. Lifecycle Check: Pre-Launch
            elif launch_date and d < launch_date:
                forecast_rows[idx] = {
                    "Date": d,
                    "SKU_ID": sku,
                    "Region": reg,
                    "Product_Family": row.get("Product_Family", meta.get("Product_Family", "")),
                    "Business_Unit": row.get("Business_Unit", meta.get("Business_Unit", "")),
                    "SKU_Description": row.get("SKU_Description", meta.get("SKU_Description", "")),
                    "Customer_Name": row.get("Customer_Name", ""),
                    "Sales_Channel": row.get("Sales_Channel", ""),
                    "horizon": h,
                    "Units_Sold_Forecast": 0,
                    "Unit_Price_USD": unit_price,
                    "Projected_Revenue_USD": 0.0,
                    "Model_Or_Fallback": "lifecycle_pre_launch",
                    "Lifecycle_Status": "Planned Launch",
                }
            # 3. Active Lifecycle
            else:
                if (sku, reg) in observed_series:
                    batch_keys.append((sku, reg))
                    batch_indices.append(idx)
                    batch_models.append(winning_model_name)
                elif replaces_sku and (replaces_sku, reg) in observed_series:
                    batch_keys.append((replaces_sku, reg))
                    batch_indices.append(idx)
                    batch_models.append(f"predecessor_{replaces_sku}_{winning_model_name}")
                else:
                    fam = meta.get("Product_Family", row.get("Product_Family", ""))
                    fb_units = family_averages.get(fam, 10)
                    forecast_rows[idx] = {
                        "Date": d,
                        "SKU_ID": sku,
                        "Region": reg,
                        "Product_Family": row.get("Product_Family", meta.get("Product_Family", "")),
                        "Business_Unit": row.get("Business_Unit", meta.get("Business_Unit", "")),
                        "SKU_Description": row.get("SKU_Description", meta.get("SKU_Description", "")),
                        "Customer_Name": row.get("Customer_Name", ""),
                        "Sales_Channel": row.get("Sales_Channel", ""),
                        "horizon": h,
                        "Units_Sold_Forecast": fb_units,
                        "Unit_Price_USD": unit_price,
                        "Projected_Revenue_USD": round(fb_units * unit_price, 2),
                        "Model_Or_Fallback": "family_peer_fallback",
                        "Lifecycle_Status": "Active",
                    }

        if batch_keys:
            preds = selected_model.predict_batch(batch_keys, d, h)
            for orig_idx, p_val, m_name in zip(batch_indices, preds, batch_models):
                row = forecast_records[orig_idx]
                sku = str(row["SKU_ID"]).strip()
                reg = str(row["Region"]).strip()
                meta = sku_meta.get(sku, {})
                base_price = base_price_by_sku.get(sku, last_prices.get(sku, 100.0))
                unit_price = round(base_price * price_multiplier, 2)
                final_units = max(0, int(round(p_val)))
                forecast_rows[orig_idx] = {
                    "Date": d,
                    "SKU_ID": sku,
                    "Region": reg,
                    "Product_Family": row.get("Product_Family", meta.get("Product_Family", "")),
                    "Business_Unit": row.get("Business_Unit", meta.get("Business_Unit", "")),
                    "SKU_Description": row.get("SKU_Description", meta.get("SKU_Description", "")),
                    "Customer_Name": row.get("Customer_Name", ""),
                    "Sales_Channel": row.get("Sales_Channel", ""),
                    "horizon": h,
                    "Units_Sold_Forecast": final_units,
                    "Unit_Price_USD": unit_price,
                    "Projected_Revenue_USD": round(final_units * unit_price, 2),
                    "Model_Or_Fallback": m_name,
                    "Lifecycle_Status": "Active",
                }

    return pd.DataFrame(forecast_rows)


def derive_submission(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Extract exactly the 4 required columns in exact test row order."""
    cols = ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    sub_df = forecast_df[cols].copy()
    sub_df["Units_Sold_Forecast"] = sub_df["Units_Sold_Forecast"].astype(int)
    return sub_df


def validate_submission(
    submission_df: pd.DataFrame,
    forecast_keys_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> SubmissionValidationReport:
    """Strictly validate submission CSV against competition and design requirements."""
    expected_cols = ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    cols = list(submission_df.columns)
    total_rows = len(submission_df)
    expected_rows = len(forecast_keys_df)

    notes: List[str] = []
    is_valid = True

    # 1. Column check
    if cols != expected_cols:
        notes.append(f"Column mismatch: got {cols}, expected {expected_cols}")
        is_valid = False
    else:
        notes.append("Column order matches exactly: Date, SKU_ID, Region, Units_Sold_Forecast")

    # 2. Row count check
    if total_rows != expected_rows:
        notes.append(f"Row count mismatch: got {total_rows:,}, expected {expected_rows:,}")
        is_valid = False
    else:
        notes.append(f"Row count verified: {total_rows:,} rows")

    # 3. Key sequence and multiplicity
    sub_keys = list(zip(submission_df["Date"], submission_df["SKU_ID"], submission_df["Region"]))
    test_keys = list(zip(forecast_keys_df["Date"], forecast_keys_df["SKU_ID"], forecast_keys_df["Region"]))
    keys_match = (sub_keys == test_keys)
    if not keys_match:
        notes.append("Key sequence or multiplicity does NOT match test file!")
        is_valid = False
    else:
        notes.append("Key sequence and multiplicity match test file identically.")

    # 4. Nulls check
    null_count = int(submission_df.isna().sum().sum())
    has_nulls = null_count > 0
    if has_nulls:
        notes.append(f"Submission contains {null_count} null/blank values!")
        is_valid = False
    else:
        notes.append("No nulls or blank values found.")

    # 5. Non-negative integer units check
    units = submission_df["Units_Sold_Forecast"]
    all_non_neg = bool((units >= 0).all())
    all_ints = bool(np.issubdtype(units.dtype, np.integer))

    if not all_non_neg:
        notes.append("Negative forecast units found!")
        is_valid = False
    if not all_ints:
        notes.append("Units_Sold_Forecast is not an integer type!")
        is_valid = False

    # 6. Agreement with published forecast
    forecast_match = bool((submission_df["Units_Sold_Forecast"] == forecast_df["Units_Sold_Forecast"]).all())
    if not forecast_match:
        notes.append("Values in submission do NOT match forecast.parquet!")
        is_valid = False
    else:
        notes.append("100% value agreement between submission.csv and forecast.parquet.")

    total_units = int(units.sum())
    total_rev = float(forecast_df["Projected_Revenue_USD"].sum())

    return SubmissionValidationReport(
        is_valid=is_valid,
        total_rows=total_rows,
        expected_rows=expected_rows,
        columns=cols,
        expected_columns=expected_cols,
        has_nulls=has_nulls,
        null_count=null_count,
        all_non_negative=all_non_neg,
        all_integers=all_ints,
        keys_match_exactly=keys_match,
        forecast_agreement=forecast_match,
        total_units_forecast=total_units,
        total_projected_revenue=total_rev,
        notes=notes,
    )
