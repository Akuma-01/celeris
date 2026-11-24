"""Robustness, edge cases, and regression tests for Celeris Demand Forecasting."""

import numpy as np
import pandas as pd
import pytest

from celeris.clean import clean_history_transactions
from celeris.demand import reconstruct_demand_and_anomalies
from celeris.features import compute_intermittency_class, extract_origin_features
from celeris.forecast import (
    SeasonalNaiveModel,
    TrailingMeanModel,
    PooledDirectModel,
    CrostonSBAModel,
    IntermittentChallengerModel,
    EnsembleModel,
)
from celeris.publish import derive_submission, validate_submission
from celeris.macro import analyze_macro_lead_lag, match_scoped_events
from app import filter_dataframe


def test_clean_empty_history():
    """Verify clean_history_transactions handles empty DataFrames without error."""
    empty_df = pd.DataFrame(columns=["Record_ID", "Date", "SKU_ID", "Region", "Units_Sold", "Unit_Price_USD", "Revenue_USD", "Load_Timestamp"])
    res_df, quality = clean_history_transactions(empty_df)
    assert res_df.empty
    assert quality == []


def test_clean_dirty_dates_normalization():
    """Verify full dates like 2020-05-15 are normalized to monthly grain 2020-05 and logged."""
    dirty_df = pd.DataFrame([
        {
            "Record_ID": "REC-01",
            "Date": "2020-05-15",
            "Year": 2020,
            "Month": 5,
            "SKU_ID": "BSS-001",
            "Region": "Europe",
            "Units_Sold": 10,
            "Unit_Price_USD": 100.0,
            "Revenue_USD": 1000.0,
            "Load_Timestamp": "2020-05-15 10:00:00",
        }
    ])
    cleaned_df, quality = clean_history_transactions(dirty_df)
    assert cleaned_df["Date"].iloc[0] == "2020-05"
    assert any(q.rule == "date_format_normalization" for q in quality)


def test_reconstruct_demand_empty_history():
    """Verify reconstruct_demand_and_anomalies handles empty history gracefully."""
    empty_df = pd.DataFrame(columns=["Date", "SKU_ID", "Region", "Units_Sold"])
    sku_master = pd.DataFrame()
    panel_df, anomalies = reconstruct_demand_and_anomalies(empty_df, sku_master)
    assert panel_df.empty
    assert anomalies == []


def test_all_models_polymorphic_predict_signature():
    """Verify all 5 models accept uniform (sku_id, region, target_date, horizon) signature."""
    dates = [f"2020-{m:02d}" for m in range(1, 13)]
    df = pd.DataFrame({
        "Date": dates,
        "Month": list(range(1, 13)),
        "SKU_ID": ["SKU-A"] * 12,
        "Region": ["North America"] * 12,
        "Product_Family": ["Brake & Safety Systems"] * 12,
        "reconstructed_demand": [10, 20, 15, 0, 25, 30, 0, 10, 15, 20, 25, 30],
        "is_observed": [True] * 12,
    })

    models = [
        SeasonalNaiveModel().fit(df, "2020-12"),
        TrailingMeanModel().fit(df, "2020-12"),
        CrostonSBAModel().fit(df, "2020-12"),
        PooledDirectModel().fit(df, "2020-12"),
        IntermittentChallengerModel().fit(df, "2020-12"),
        EnsembleModel().fit(df, "2020-12"),
    ]

    for m in models:
        # Uniform call with 4 positional arguments
        pred = m.predict("SKU-A", "North America", "2021-01", 1)
        assert isinstance(pred, int)
        assert pred >= 0

        # Also works with 3 arguments (horizon default)
        pred_def = m.predict("SKU-A", "North America", "2021-01")
        assert isinstance(pred_def, int)
        assert pred_def >= 0


def test_validate_submission_catches_invalid_formats():
    """Verify validate_submission catches negative units, nulls, and key mismatches."""
    keys_df = pd.DataFrame({
        "Date": ["2026-11", "2026-12"],
        "SKU_ID": ["SKU-1", "SKU-1"],
        "Region": ["Europe", "Europe"],
    })
    forecast_df = pd.DataFrame({
        "Date": ["2026-11", "2026-12"],
        "SKU_ID": ["SKU-1", "SKU-1"],
        "Region": ["Europe", "Europe"],
        "Units_Sold_Forecast": [10, 20],
        "Projected_Revenue_USD": [1000.0, 2000.0],
    })

    # 1. Valid submission passes
    valid_sub = derive_submission(forecast_df)
    rep = validate_submission(valid_sub, keys_df, forecast_df)
    assert rep.is_valid is True

    # 2. Negative forecast units caught
    neg_sub = valid_sub.copy()
    neg_sub.loc[0, "Units_Sold_Forecast"] = -5
    rep_neg = validate_submission(neg_sub, keys_df, forecast_df)
    assert rep_neg.is_valid is False
    assert rep_neg.all_non_negative is False

    # 3. Null values caught
    null_sub = valid_sub.copy()
    null_sub.loc[0, "Units_Sold_Forecast"] = None
    rep_null = validate_submission(null_sub, keys_df, forecast_df)
    assert rep_null.is_valid is False
    assert rep_null.has_nulls is True

    # 4. Key order mismatch caught
    rev_sub = valid_sub.iloc[::-1].reset_index(drop=True)
    rep_rev = validate_submission(rev_sub, keys_df, forecast_df)
    assert rep_rev.is_valid is False
    assert rep_rev.keys_match_exactly is False


def test_macro_lead_lag_handles_constant_series():
    """Verify analyze_macro_lead_lag does not fail or produce NaN on constant demand."""
    dates = [f"2020-{m:02d}" for m in range(1, 13)]
    macro_df = pd.DataFrame({
        "Date": dates,
        "ISM_Manufacturing_PMI": [50.0] * 12,
    })
    history_df = pd.DataFrame({
        "Date": dates,
        "Product_Family": ["Family_A"] * 12,
        "reconstructed_demand": [100.0] * 12,  # zero variance
    })

    results = analyze_macro_lead_lag(history_df, macro_df)
    assert "Family_A" in results
    res = results["Family_A"]
    assert not np.isnan(res.correlation)
    assert res.correlation == 0.0


def test_filter_dataframe_empty_cases():
    """Verify filter_dataframe returns empty gracefully on non-matching filters."""
    df = pd.DataFrame({
        "Product_Family": ["BSS", "HYD"],
        "SKU_ID": ["SKU-1", "SKU-2"],
        "Region": ["Europe", "North America"],
    })

    # Non-matching filter
    filtered = filter_dataframe(df, {"Region": ["NonExistentRegion"]})
    assert filtered.empty

    # "All" returns unfiltered
    unfiltered = filter_dataframe(df, {"Region": ["All"]})
    assert len(unfiltered) == 2
