"""Tests for Celeris Advanced Tier features:
- Intermittent demand forecasting (Croston SBA & Intermittent Challenger)
- Macroeconomic lead/lag analysis
- Scoped calendar events
- Advanced artifacts publication (macro_lead_lag, stockouts, sku_master)
- Dashboard View 6 & 7 data integrity
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from celeris.forecast import CrostonSBAModel, IntermittentChallengerModel
from celeris.macro import analyze_macro_lead_lag, match_scoped_events
from celeris.pipeline import PipelineRunner
from app import load_data, filter_dataframe


def test_croston_sba_model_basic():
    """Verify Croston SBA fits sporadic series and applies debiasing factor."""
    dates = [f"2020-{m:02d}" for m in range(1, 13)]
    # Series with positive demands at month 2 (10 units), month 5 (20 units), month 10 (15 units)
    demands = [0, 10, 0, 0, 20, 0, 0, 0, 0, 15, 0, 0]
    df = pd.DataFrame({
        "Date": dates,
        "SKU_ID": ["SKU-INT"] * 12,
        "Region": ["Europe"] * 12,
        "reconstructed_demand": demands,
    })

    model = CrostonSBAModel(alpha=0.1)
    model.fit(df, origin_date="2020-12")

    pred = model.predict("SKU-INT", "Europe", "2021-01")
    assert isinstance(pred, int)
    assert pred >= 0

    # Test unknown series returns 0
    assert model.predict("SKU-UNKNOWN", "Europe", "2021-01") == 0


def test_intermittent_challenger_dispatch():
    """Verify IntermittentChallengerModel identifies sporadic series and dispatches correctly."""
    dates = [f"2020-{m:02d}" for m in range(1, 13)]
    df = pd.DataFrame({
        "Date": dates * 2,
        "SKU_ID": ["SKU-SPORADIC"] * 12 + ["SKU-SMOOTH"] * 12,
        "Region": ["Europe"] * 24,
        "Product_Family": ["Hydraulic Systems"] * 24,
        "Sales_Channel": ["OEM Direct"] * 24,
        "Customer_Name": ["EuroTruck AG"] * 24,
        "is_observed": [True] * 24,
        "reconstructed_demand": [0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 12, 0] + [100, 105, 110, 95, 102, 108, 98, 101, 104, 107, 103, 100],
    })

    model = IntermittentChallengerModel(alpha=0.1, random_state=42)
    model.fit(df, origin_date="2020-12")

    pred_sporadic = model.predict("SKU-SPORADIC", "Europe", "2021-01", horizon=1)
    pred_smooth = model.predict("SKU-SMOOTH", "Europe", "2021-01", horizon=1)

    assert isinstance(pred_sporadic, int)
    assert isinstance(pred_smooth, int)
    assert pred_sporadic >= 0
    assert pred_smooth > 50  # Smooth series handled by pooled regressor


def test_analyze_macro_lead_lag():
    """Verify lead/lag cross-correlation across lags 0 to 12."""
    dates = [f"{y}-{m:02d}" for y in range(2020, 2023) for m in range(1, 13)]
    n = len(dates)

    # Synthetic macro indicator: sine wave
    t = np.arange(n)
    pmi = 50.0 + 5.0 * np.sin(2 * np.pi * t / 12)
    macro_df = pd.DataFrame({
        "Date": dates,
        "ISM_Manufacturing_PMI": pmi,
    })

    # Synthetic family demand: leading indicator with 3 months lag
    # demand(t) = PMI(t - 3)
    pmi_lag3 = np.roll(pmi, 3)
    pmi_lag3[:3] = 50.0
    history_df = pd.DataFrame({
        "Date": dates,
        "Product_Family": ["Brake & Safety Systems"] * n,
        "reconstructed_demand": pmi_lag3 * 10,
    })

    results = analyze_macro_lead_lag(history_df, macro_df, max_lag=12)
    assert "Brake & Safety Systems" in results
    res = results["Brake & Safety Systems"]
    assert res.best_indicator == "ISM_Manufacturing_PMI"
    assert res.best_lag in [2, 3, 4]  # Peak correlation near lag 3
    assert res.correlation > 0.85


def test_scoped_event_matching():
    """Verify hierarchical scope matching for calendar events."""
    events_df = pd.DataFrame([
        {
            "Event_ID": "EV-001",
            "Event_Type": "Annual Price Change",
            "Start_Date": "2023-01",
            "End_Date": "2023-01",
            "Scope_Type": "Global",
            "Scope_Value": "All",
        },
        {
            "Event_ID": "EV-002",
            "Event_Type": "Supply Disruption",
            "Start_Date": "2023-06",
            "End_Date": "2023-07",
            "Scope_Type": "Region",
            "Scope_Value": "Europe",
        },
    ])

    m1 = match_scoped_events(events_df, "2023-01", "BSS-001", "North America", "BSS", "OEM", "CustA")
    assert m1["has_event"] is True
    assert m1["is_price_change"] is True
    assert "EV-001" in m1["active_event_ids"]

    m2 = match_scoped_events(events_df, "2023-06", "BSS-001", "Europe", "BSS", "OEM", "CustA")
    assert m2["has_event"] is True
    assert m2["is_supply_constraint"] is True
    assert "EV-002" in m2["active_event_ids"]

    m3 = match_scoped_events(events_df, "2023-06", "BSS-001", "North America", "BSS", "OEM", "CustA")
    assert m3["has_event"] is False


def test_pipeline_publishes_advanced_artifacts(tmp_path):
    """Verify full pipeline execution materializes macro_lead_lag, stockouts, sku_master, and 4 backtest candidates."""
    out_dir = tmp_path / "advanced_run"
    runner = PipelineRunner(data_dir="fixtures/data", out_dir=out_dir, is_fixture=True)
    res = runner.run()

    assert res.success is True
    assert (out_dir / "macro_lead_lag.parquet").exists()
    assert (out_dir / "stockouts.parquet").exists()
    assert (out_dir / "sku_master.parquet").exists()

    macro_df = pd.read_parquet(out_dir / "macro_lead_lag.parquet")
    assert not macro_df.empty
    assert "Product_Family" in macro_df.columns
    assert "Lag_Months" in macro_df.columns
    assert "Correlation" in macro_df.columns

    bt_df = pd.read_parquet(out_dir / "backtest.parquet")
    candidates = bt_df["candidate"].unique().tolist()
    assert "seasonal_naive" in candidates
    assert "trailing_mean" in candidates
    assert "pooled_direct" in candidates
    assert "intermittent_challenger" in candidates

    # Verify manifest metadata
    with open(out_dir / "manifest.json", "r") as f:
        manifest_data = json.load(f)
    assert manifest_data["metadata"]["tier"] == "Advanced"
    assert "macro_leading_indicators" in manifest_data["diagnostics_summary"]


def test_dashboard_views_load_advanced_artifacts(tmp_path):
    """Verify app.py loads advanced artifacts cleanly."""
    out_dir = tmp_path / "dash_advanced"
    runner = PipelineRunner(data_dir="fixtures/data", out_dir=out_dir, is_fixture=True)
    runner.run()

    data = load_data(str(out_dir))
    assert "macro_lead_lag" in data
    assert "stockouts" in data
    assert "sku_master" in data
    assert not data["stockouts"].empty
    assert not data["sku_master"].empty
