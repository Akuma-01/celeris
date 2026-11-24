"""Tests for Milestone 5: Interactive dashboard logic, view integrity, and row reconciliation."""

from pathlib import Path
import pandas as pd
import pytest

from app import load_data, filter_dataframe


@pytest.fixture
def published_data():
    """Load published fixture run artifacts."""
    return load_data("artifacts/fixture_run")


def test_dashboard_data_loading(published_data):
    """Verify that the dashboard loads all required tables from published artifacts."""
    assert not published_data["history"].empty
    assert not published_data["forecast"].empty
    assert not published_data["quality"].empty
    assert not published_data["anomalies"].empty
    assert not published_data["backtest"].empty
    assert published_data["manifest"]


def test_filtered_totals_reconciliation_unfiltered(published_data):
    """Verify that unfiltered dashboard totals match underlying published rows exactly."""
    forecast_df = published_data["forecast"]
    history_df = published_data["history"]

    filters = {"Product_Family": ["All"], "SKU_ID": ["All"], "Region": ["All"]}

    filt_forecast = filter_dataframe(forecast_df, filters)
    filt_history = filter_dataframe(history_df, filters)

    # 1. Forecast total units
    dashboard_forecast_units = int(filt_forecast["Units_Sold_Forecast"].sum())
    underlying_forecast_units = int(forecast_df["Units_Sold_Forecast"].sum())
    assert dashboard_forecast_units == underlying_forecast_units
    assert dashboard_forecast_units == 5480  # exact fixture sum

    # 2. Forecast total projected revenue
    dashboard_revenue = round(float(filt_forecast["Projected_Revenue_USD"].sum()), 2)
    underlying_revenue = round(float(forecast_df["Projected_Revenue_USD"].sum()), 2)
    assert dashboard_revenue == underlying_revenue

    # 3. Cleaned historical shipments
    dashboard_hist_units = int(filt_history["Units_Sold"].sum(skipna=True))
    underlying_hist_units = int(history_df["Units_Sold"].sum(skipna=True))
    assert dashboard_hist_units == underlying_hist_units

    # 4. Reconstructed demand
    dashboard_reconstructed = float(filt_history["reconstructed_demand"].sum(skipna=True))
    underlying_reconstructed = float(history_df["reconstructed_demand"].sum(skipna=True))
    assert dashboard_reconstructed == underlying_reconstructed


def test_filtered_totals_reconciliation_by_region(published_data):
    """Verify that a region-filtered dashboard total matches underlying rows identically."""
    forecast_df = published_data["forecast"]
    history_df = published_data["history"]

    filters = {"Region": ["North America"]}
    filt_forecast = filter_dataframe(forecast_df, filters)
    filt_history = filter_dataframe(history_df, filters)

    # Direct pandas slice for ground truth
    true_fct_slice = forecast_df[forecast_df["Region"] == "North America"]
    true_hist_slice = history_df[history_df["Region"] == "North America"]

    # Assert exact row count and totals
    assert len(filt_forecast) == len(true_fct_slice)
    assert int(filt_forecast["Units_Sold_Forecast"].sum()) == int(true_fct_slice["Units_Sold_Forecast"].sum())
    assert round(float(filt_forecast["Projected_Revenue_USD"].sum()), 2) == round(float(true_fct_slice["Projected_Revenue_USD"].sum()), 2)

    assert len(filt_history) == len(true_hist_slice)
    assert int(filt_history["Units_Sold"].sum(skipna=True)) == int(true_hist_slice["Units_Sold"].sum(skipna=True))


def test_filtered_totals_reconciliation_by_sku(published_data):
    """Verify that an SKU-filtered dashboard total matches underlying rows identically."""
    forecast_df = published_data["forecast"]
    history_df = published_data["history"]
    anomalies_df = published_data["anomalies"]

    filters = {"SKU_ID": ["BSS-001"]}
    filt_forecast = filter_dataframe(forecast_df, filters)
    filt_history = filter_dataframe(history_df, filters)
    filt_anomalies = filter_dataframe(anomalies_df, filters)

    true_fct_slice = forecast_df[forecast_df["SKU_ID"] == "BSS-001"]
    true_hist_slice = history_df[history_df["SKU_ID"] == "BSS-001"]
    true_anom_slice = anomalies_df[anomalies_df["SKU_ID"] == "BSS-001"]

    assert len(filt_forecast) == len(true_fct_slice)
    assert int(filt_forecast["Units_Sold_Forecast"].sum()) == int(true_fct_slice["Units_Sold_Forecast"].sum())
    assert len(filt_history) == len(true_hist_slice)
    assert len(filt_anomalies) == len(true_anom_slice)


def test_real_run_artifacts_loading_and_reconciliation():
    """Verify that dashboard load_data loads and reconciles real_run artifacts cleanly if present."""
    real_path = Path("artifacts/real_run")
    if not real_path.exists():
        pytest.skip("artifacts/real_run does not exist")

    data = load_data("artifacts/real_run")
    assert not data["history"].empty
    assert not data["forecast"].empty
    assert not data["quality"].empty
    assert not data["anomalies"].empty
    assert not data["backtest"].empty
    assert not data["stockouts"].empty
    assert not data["sku_master"].empty
    assert not data["macro_lead_lag"].empty
    assert data["manifest"]["run_mode"] == "real"

    # Reconciliation check
    forecast_df = data["forecast"]
    assert len(forecast_df) == 58219
    sub_df = pd.read_csv("artifacts/real_run/submission.csv")
    assert int(forecast_df["Units_Sold_Forecast"].sum()) == int(sub_df["Units_Sold_Forecast"].sum())
