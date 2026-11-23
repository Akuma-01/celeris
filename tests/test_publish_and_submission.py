"""Tests for Milestone 4: Forecast publication, lifecycle zero enforcement, and submission verification."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from celeris.pipeline import PipelineRunner
from celeris.publish import validate_submission, derive_submission


def test_forecast_and_submission_pipeline_run(fixture_data_dir: Path, tmp_path: Path):
    """Run pipeline on fixture and assert all Milestone 4 requirements are satisfied."""
    runner = PipelineRunner(data_dir=fixture_data_dir, out_dir=tmp_path / "pub_test", is_fixture=True)
    res = runner.run()

    assert res.success is True
    assert res.status == "SUCCESS"

    # Check published files exist
    forecast_path = tmp_path / "pub_test" / "forecast.parquet"
    submission_path = tmp_path / "pub_test" / "submission.csv"

    assert forecast_path.exists()
    assert submission_path.exists()

    forecast_df = pd.read_parquet(forecast_path)
    submission_df = pd.read_csv(submission_path)
    test_keys_df = pd.read_csv(fixture_data_dir / "forecast_rows.csv")

    # 1. Exact columns and order in submission.csv
    expected_cols = ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    assert list(submission_df.columns) == expected_cols

    # 2. Row count matches test keys exactly
    assert len(submission_df) == len(test_keys_df)
    assert len(submission_df) == 76  # 4 SKUs x 19 months in fixture

    # 3. Exact key order and multiplicity
    sub_keys = list(zip(submission_df["Date"], submission_df["SKU_ID"], submission_df["Region"]))
    test_keys = list(zip(test_keys_df["Date"], test_keys_df["SKU_ID"], test_keys_df["Region"]))
    assert sub_keys == test_keys

    # 4. No nulls or blank values
    assert submission_df.isna().sum().sum() == 0

    # 5. Non-negative integers
    units = submission_df["Units_Sold_Forecast"]
    assert (units >= 0).all()
    assert np.issubdtype(units.dtype, np.integer)

    # 6. Agreement between submission.csv and forecast.parquet
    assert (submission_df["Units_Sold_Forecast"] == forecast_df["Units_Sold_Forecast"]).all()

    # 7. Lifecycle constraint: HYD-101 is discontinued (EOL: 2024-12)
    # In forecast window (2026-11 -> 2028-05), ALL HYD-101 forecasts MUST BE ZERO!
    hyd_101_fct = forecast_df[forecast_df["SKU_ID"] == "HYD-101"]
    assert len(hyd_101_fct) == 19
    assert (hyd_101_fct["Units_Sold_Forecast"] == 0).all()
    assert (hyd_101_fct["Model_Or_Fallback"] == "lifecycle_post_eol").all()
    assert (hyd_101_fct["Projected_Revenue_USD"] == 0.0).all()

    # Active SKUs have positive forecasts
    active_fct = forecast_df[forecast_df["SKU_ID"].isin(["BSS-001", "HYD-102", "PWR-201"])]
    assert (active_fct["Units_Sold_Forecast"] > 0).all()
    assert (active_fct["Projected_Revenue_USD"] > 0.0).all()

    # 8. Validation report passed
    val = res.submission_validation
    assert val is not None
    assert val.is_valid is True
    assert val.all_integers is True
    assert val.all_non_negative is True
    assert val.keys_match_exactly is True
    assert val.forecast_agreement is True
